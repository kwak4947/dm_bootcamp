#include "dm_controller/controller.h"

namespace DmController {
    Controller::Controller(const std::string urdf_path) : urdf_path_(urdf_path) {
        std::cout << "Robot Controller is ready to be initialized" << std::endl;
        initialize(urdf_path);
        tau_d_lpf_.setZero();
        f_ext_lpf_.setZero();
        v_lpf_.setZero();
        v_ee_lpf_.setZero();
    }

    void Controller::initialize(const std::string urdf_path) {
        model_.initialize(urdf_path);
        q_desired_.setZero();
        q_init_.setZero();
        gw_desired_.setZero();
        gw_init_.setZero();
        q_idle_ << 0.0, -0.2, 0, -1.57, 0, 1.57, 0.71;
        gw_idle_ << 0.04, 0.04;
        t_ = 0.0;
        set_init_ = true;
    }

    void Controller::updateModel(const Eigen::Ref<const Eigen::VectorXd>& q,
                                 const Eigen::Ref<const Eigen::VectorXd>& qd,
                                 const Eigen::Ref<const Eigen::VectorXd>& tau,
                                 const Eigen::Ref<const Eigen::VectorXd>& ft,
                                 const double t) {
        if (!q.allFinite() || !qd.allFinite() || !tau.allFinite() || !ft.allFinite()) {
            std::cout << "🧨 NaN detected in updateModel input!" << std::endl;
            std::cout << "q: " << q.transpose() << std::endl;
            std::cout << "qd: " << qd.transpose() << std::endl;
            std::cout << "tau: " << tau.transpose() << std::endl;
            std::cout << "ft: " << ft.transpose() << std::endl;
        }

        t_ = t / hz_;
        q_ = q.head<7>();
        qd_ = qd.head<7>();
        tau_measured_ = tau;
        f_ext_ = ft;

        double f_cut = 30;
        double ts = 1 / (2 * M_PI * f_cut);
        f_ext_lpf_ = dyros_math::lowPassFilter(f_ext_, f_ext_lpf_, 0.001, ts);

        gw_ = q.tail<2>();
        gv_ = qd.tail<2>();

        model_.updateModel(q_, qd_, gw_, tau_measured_);
        x_ = model_.p_;
        v_ = model_.xd_;

        f_cut = 60;
        ts = 1 / (2 * M_PI * f_cut);
        v_lpf_ = dyros_math::lowPassFilter(v_, v_lpf_, 1 / hz_, ts);

        R_ = model_.r_;
        tau_ext_ = model_.tau_ext_;
    }

    Eigen::VectorXd Controller::setIdleConfig(const Eigen::Ref<const Eigen::Vector2d>& time) {
        Eigen::Vector7d tau_d;
        for (int i = 0; i < dof_; i++) {
            q_desired_(i) = cubic(t_, time(0), time(1), model_.q_init_(i), q_idle_(i), 0.0, 0.0);
        }
        tau_d = 500 * (q_desired_ - q_) - 10 * qd_;
        tau_d += model_.NLE_;
        return tau_d;
    }

    Eigen::VectorXd Controller::gripperOpen(const double target_width, const Eigen::Ref<const Eigen::Vector2d>& time) {
        Eigen::Vector2d gw_init = model_.gw_init_;
        Eigen::Vector2d gf;
        double t_0 = time(0), t_f = time(1);
        for (int i = 0; i < 2; i++) {
            gw_desired_(i) = cubic(t_, t_0, t_f, gw_init(i), target_width, 0.0, 0.0);
        }
        gf = 100.0 * (gw_desired_ - gw_) - 10.0 * gv_;
        return gf;
    }

    Eigen::VectorXd Controller::gripperClose() {
        Eigen::Vector2d gf;
        for (int i = 0; i < 2; i++) gf(i) = -5.0 - 50 * gv_(i);
        return gf;
    }

    Eigen::VectorXd Controller::taskMove(const Eigen::Ref<const Eigen::Vector3d> &x_goal,
                                        const Eigen::Ref<const Eigen::MatrixXd> &r_goal,
                                        const Eigen::Ref<const Eigen::Vector2d> &time)
    {
        Eigen::Vector3d x_0, x_f;
        Eigen::Vector3d xd_ddot, xd_dot, xd;
        Eigen::Vector3d xd_set;
        Eigen::Matrix3d R_0, R_f, Rd;
        Eigen::Vector6d f;
        Eigen::Vector7d tau_d, tau_nll;
        double t_0, t_f;

        x_0 = model_.initial_transform_.translation();
        x_f = model_.initial_transform_.linear() * x_goal + x_0;

        R_0 = model_.initial_transform_.linear();
        R_f = model_.initial_transform_.linear() * r_goal;

        t_0 = time(0);
        t_f = time(1);
        for (int i = 0; i < 3; i++) {
            xd_set = dyros_math::quinticSpline(t_, t_0, t_f, x_0(i), 0.0, 0.0, x_f(i), 0.0, 0.0);
            xd(i) = xd_set(0);
            xd_dot(i) = xd_set(1);
            xd_ddot(i) = xd_set(2);
        }

        Rd = dyros_math::rotationCubic(t_, t_0, t_f, R_0, R_f);

        f.head<3>() = xd_ddot + (2000 * (xd - x_) + 80 * (xd_dot - v_lpf_.head<3>()));
        f.tail<3>() = 3000 * (-dyros_math::getPhi(R_, Rd)) + 60 * (-v_.tail<3>());

        tau_nll = 100.0 * (model_.q_init_ - q_) - 10.0 * (qd_);

        // 디버깅: force vector와 각 요소 체크
        if (!model_.J_.allFinite()) std::cout << "🧨 model_.J_ contains NaN/Inf\n";
        if (!model_.A_.allFinite()) std::cout << "🧨 model_.A_ contains NaN/Inf\n";
        if (!f.allFinite()) std::cout << "🧨 force vector f contains NaN/Inf: " << f.transpose() << "\n";
        if (!model_.NLE_.allFinite()) std::cout << "🧨 NLE_ contains NaN/Inf\n";
        if (!model_.N_.allFinite()) std::cout << "🧨 N matrix contains NaN/Inf\n";
        if (!tau_nll.allFinite()) std::cout << "🧨 tau_nll contains NaN/Inf: " << tau_nll.transpose() << "\n";

        Eigen::Vector6d tau_f = model_.A_ * f;
        if (!tau_f.allFinite()) {
            std::cout << "🧨 model_.A_ * f produced NaN\n";
            std::cout << "A: " << model_.A_ << "\nf: " << f.transpose() << std::endl;
            exit(1);
        }

        Eigen::Vector7d tau_task = model_.J_.transpose() * tau_f;
        if (!tau_task.allFinite()) {
            std::cout << "🧨 J^T * (A*f) produced NaN\n";
            std::cout << "J^T: " << model_.J_.transpose() << "\ntau_f: " << tau_f.transpose() << std::endl;
            exit(1);
        }

        tau_d = tau_task + model_.NLE_ + model_.N_ * tau_nll;

        if (!tau_d.allFinite()) {
            std::cout << "[🧨 taskMove] tau_d has NaN! " << tau_d.transpose() << std::endl;
            std::cout << "tau_task: " << tau_task.transpose() << std::endl;
            std::cout << "NLE: " << model_.NLE_.transpose() << std::endl;
            std::cout << "N: " << model_.N_ << std::endl;
            std::cout << "tau_nll: " << tau_nll.transpose() << std::endl;
            exit(1);
        }

        return tau_d;
    }

    Eigen::VectorXd Controller::jointMove(const Eigen::Ref<const Eigen::VectorXd>& goal,
                                          const Eigen::Ref<const Eigen::Vector2d>& time) {
        Eigen::Vector7d tau_d;
        for (int i = 0; i < dof_; i++) {
            q_desired_(i) = cubic(t_, time(0), time(1), model_.q_init_(i), goal(i), 0.0, 0.0);
        }
        tau_d = 500 * (q_desired_ - q_) - 10 * qd_;
        tau_d += model_.NLE_;
        // std::cout << "jointmovejointmovejointmove" << std::endl;
        return tau_d;
    }

    Eigen::VectorXd Controller::idleState() {
        Eigen::Vector8d ctrl;
        tau_d_ = 500 * (q_idle_ - q_) - 10 * qd_;
        tau_d_ += model_.NLE_;
        gf_ = 50 * (gw_idle_ - gw_) - 10 * gv_;
        ctrl.head<7>() = tau_d_;
        ctrl.tail<1>() = gf_.head<1>();
        return ctrl;
    }

    Eigen::VectorXd Controller::initState() {
        Eigen::Vector8d ctrl;
        tau_d_ = 500 * (model_.q_init_ - q_) - 10 * qd_;
        tau_d_ += model_.NLE_;
        gf_ = 50 * (model_.gw_init_ - gw_) - 10 * gv_;
        ctrl.head<7>() = tau_d_;
        ctrl.tail<1>() = gf_.head<1>();
        return ctrl;
    }

    Eigen::VectorXd Controller::getIdleConfig() const {
        return q_idle_;
    }

    void Controller::updateInitialValues() {
        model_.initial_transform_.translation() = x_;
        model_.initial_transform_.linear() = R_;
        model_.q_init_ = q_;
        model_.gw_init_ = gw_;
    }
}
