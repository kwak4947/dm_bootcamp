import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer as mj_view
import threading
import time
from .scene_monitor import SceneMonitor
from .image_publisher import MujocoCameraBridge
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.action import ActionServer, CancelResponse, GoalResponse
import numpy as np


def clamp(value, low, high):
    return max(low, min(value, high))


class MujocoROSBridge(Node):
    def __init__(self, robot_info, camera_info, robot_controller):
        super().__init__('mujoco_ros_bridge')

        self.xml_path = robot_info[0]
        self.urdf_path = robot_info[1]
        self.ctrl_freq = robot_info[2]

        self.camera_name = camera_info[0]
        self.width = camera_info[1]
        self.height = camera_info[2]
        self.fps = camera_info[3]

        self.rc = robot_controller

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.dt = 1 / self.ctrl_freq
        self.model.opt.timestep = self.dt

        self.sm = SceneMonitor(self.model, self.data)
        self.hand_eye = MujocoCameraBridge(self.model, camera_info)

        self.ctrl_dof = 8
        self.ctrl_step = 0

        self.running = True
        self.lock = threading.Lock()
        self.robot_thread = threading.Thread(target=self.robot_control, daemon=True)
        self.hand_eye_thread = threading.Thread(target=self.hand_eye_control, daemon=True)
        self.ros_thread = threading.Thread(target=self.ros_control, daemon=True)

        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3",
            "panda_joint4", "panda_joint5", "panda_joint6",
            "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"
        ]

        self.mujoco_joint_names = [f"fr3_joint{i+1}" for i in range(7)]
        self.joint_name_map = {
            moveit: mujoco for moveit, mujoco in zip(self.joint_names[:7], self.mujoco_joint_names)
        }
        self.joint_index_map = {
            name: self.model.joint(name).qposadr for name in self.joint_name_map.values()
        }

        self.follow_trajectory_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/panda_arm_controller/follow_joint_trajectory',
            execute_callback=self.execute_trajectory_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        self.active_trajectory = None
        self.trajectory_lock = threading.Lock()
        self.current_point_index = 0
        self.trajectory_start_time = None
        self.last_target = None

        self.last_gripper_pos = 0.04
        self.initialized = False
        self.box_initial_pose = None

    def goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_trajectory_callback(self, goal_handle):
        with self.trajectory_lock:
            self.active_trajectory = goal_handle.request.trajectory
            self.current_point_index = 0
            self.trajectory_start_time = time.time()
        goal_handle.succeed()
        return FollowJointTrajectory.Result()

    def run(self):
        scene_update_freq = 30
        try:
            with mj_view.launch_passive(self.model, self.data) as viewer:
                self.robot_thread.start()
                self.hand_eye_thread.start()
                self.ros_thread.start()

                while self.running and viewer.is_running():
                    start_time = time.perf_counter()
                    with self.lock:
                        viewer.sync()
                    self.time_sync(1/scene_update_freq, start_time, False)
        except KeyboardInterrupt:
            self.running = False
            self.robot_thread.join()
            self.hand_eye_thread.join()
            self.ros_thread.join()
            self.sm.destroy_node()

    def robot_control(self):
        self.ctrl_step = 0

        try:
            while rclpy.ok() and self.running:
                with self.lock:
                    start_time = time.perf_counter()

                    if not self.initialized:
                        self.initialized = True
                        mujoco.mj_forward(self.model, self.data)

                    with self.trajectory_lock:
                        now = time.time() - self.trajectory_start_time if self.trajectory_start_time else None
                        if self.active_trajectory and now is not None:
                            points = self.active_trajectory.points
                            for i in range(len(points) - 1):
                                t0 = points[i].time_from_start.sec + points[i].time_from_start.nanosec * 1e-9
                                t1 = points[i + 1].time_from_start.sec + points[i + 1].time_from_start.nanosec * 1e-9
                                if t0 <= now <= t1:
                                    ratio = (now - t0) / (t1 - t0)
                                    q0 = np.array(points[i].positions)
                                    q1 = np.array(points[i + 1].positions)
                                    self.last_target = (1 - ratio) * q0 + ratio * q1
                                    break
                            else:
                                if now > points[-1].time_from_start.sec + points[-1].time_from_start.nanosec * 1e-9:
                                    self.last_target = np.array(points[-1].positions)
                                    self.active_trajectory = None
                                    self.trajectory_start_time = None
                                    self.rc.controller.q_init_ = self.last_target[:7]
                                    self.rc.controller.gw_init_ = np.array([0.04, 0.04])

                    if self.box_initial_pose is None:
                        self.box_initial_pose = np.copy(self.data.qpos[9:16])

                    if self.last_target is not None:
                        if len(self.last_target) >= 7:
                            for i, name in enumerate(self.joint_names[:7]):
                                if name in self.joint_name_map:
                                    mj_name = self.joint_name_map[name]
                                    mj_index = self.joint_index_map[mj_name]
                                    self.data.qpos[mj_index] = self.last_target[i]
                            self.data.qpos[7] = self.last_gripper_pos
                            self.data.qpos[8] = self.last_gripper_pos

                    if self.last_target is None:
                        self.rc.updateModel(self.data, self.ctrl_step)
                        tau = self.rc.compute()
                        self.data.ctrl[:self.ctrl_dof] = tau

                    mujoco.mj_forward(self.model, self.data)
                    mujoco.mj_step(self.model, self.data)

                    msg = JointState()
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.name = self.joint_names
                    msg.position = list(self.data.qpos[:7]) + [self.data.qpos[7], self.data.qpos[8]]
                    msg.velocity = [0.0] * len(msg.name)
                    msg.effort = [0.0] * len(msg.name)
                    self.joint_state_pub.publish(msg)

                    self.ctrl_step += 1

                self.time_sync(self.dt, start_time, False)

        except KeyboardInterrupt:
            self.rc.destroy_node()

    def hand_eye_control(self):
        renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)
        hand_eye_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)

        while rclpy.ok() and self.running:
            with self.lock:
                start_time = time.perf_counter()
                renderer.update_scene(self.data, camera=hand_eye_id)
                self.hand_eye.getImage(renderer.render(), self.ctrl_step)
            self.time_sync(1/self.fps, start_time, False)
        self.hand_eye.destroy_node()

    def time_sync(self, target_dt, t_0, verbose=False):
        elapsed_time = time.perf_counter() - t_0
        sleep_time = target_dt - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

    def ros_control(self):
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(self.rc.tm)
        executor.add_node(self.rc.jm)
        executor.add_node(self.hand_eye)
        executor.add_node(self)
        executor.spin()
        executor.shutdown()
        self.rc.tm.destroy_node()
        self.rc.jm.destroy_node()
