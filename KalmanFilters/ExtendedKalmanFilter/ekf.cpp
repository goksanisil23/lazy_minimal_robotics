#include "kalman.hpp"

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& P0) : A(A), B(B), C(C), Q(Q), R(R), P_prio(P0), P_post(P0), 
                            n(A.rows()), m(C.rows()), c(B.cols()), I(n,n),
                            initialized(false)
{
    I.setIdentity();
}

void ExtendedKalmanFilter::init(const Eigen::VectorXd& x0) {
    x_hat_prio = x0;
    x_hat_post = x0;

    initialized = true;
}

void ExtendedKalmanFilter::prediction_update(const Eigen::VectorXd& u) {
    if(!initialized) {
        throw std::runtime_error("Filter is not initialized!");
    }

    // Use the posterior estimates from the previous innovation
    x_hat_prio = A * x_hat_post + B * u;
    P_prio = A * P_post * A.transpose() + Q;
}

void ExtendedKalmanFilter::innovation_update(const Eigen::VectorXd& y) {
    K = P_prio * C.transpose() * (C * P_prio * C.transpose() + R).inverse();
    x_hat_post = x_hat_prio + K * (y - C * x_hat_prio);
    P_post = (I - K * C) * P_prio;
}