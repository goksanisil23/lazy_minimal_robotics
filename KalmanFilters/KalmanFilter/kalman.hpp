#pragma once

#include <iostream>
#include <Eigen/Dense>

class KalmanFilter {

public:

    // Generate the Kalman Filter with specified matrices
    // A : System dynamics matrix
    // B : Input matrix
    // C : Output matrix
    // Q : Process noise covariance
    // R : Measurement noise covariance
    // P0 : Initial error covariance
    KalmanFilter(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& P0);

    // Initialize the filter with a guess for the initial states
    void init(const Eigen::VectorXd& x0);

    // Update the a-priori estimates based on the system model and control input
    void prediction_update(const Eigen::VectorXd& u);

    // Update the a-posteriori estimates based on the measurements
    void innovation_update(const Eigen::VectorXd& y);

    // Return the current state
    Eigen::VectorXd get_state() {return x_hat_post;};

    // Return the current state error covariance
    Eigen::VectorXd get_P() {return P_post;};    

private:
    // Matrices used in predict and correction steps
    Eigen::MatrixXd A, B, C, Q, R, P_prio, P_post, K;

    // system dimensions
    // n : # state variables to be estimated
    // m : # measured variables
    // c : # control input variables
    int32_t m, n, c;

    // n-sized identity
    Eigen::MatrixXd I;

    // is the filter initialized?
    bool initialized = false;

    // Estimated states
    Eigen::VectorXd x_hat_prio, x_hat_post;

};
