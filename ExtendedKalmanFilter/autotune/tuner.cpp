/**
 * @brief This is a simple example usage of the Extended Kalman Filter
 * It is used to estimate the 2D position(x,y) and heading(h) of a mobile robot, by using linear velocity(v) and angular rate(w) measurements.
 *
 * X (state vector) = [x y h v]' 
 * Z (measurement vector) = [x y]
 * U (input/control vector) = [v_in w_in]'
 * 
 * Motion Model: X_(k+1) = A_(k) * X_(k) + B_(k) * U_(k) = f(X,U) 
 * x_(k+1) = x_(k) + dt * cos(h) * v_in
 * y_(k+1) = x_(k) + dt * sin(h) * v_in
 * h_(k+1) = h_(k) + dt * w_in
 * v_(k+1) = v_in
 * 
 * Measurement Model: Z_(k) = H_(k) * X_(k) = h(X)
 * x_meas = x_(k)
 * y_meas = y_(k)
 * 
 * -------------------
 * Jacobian of Motion Model : F(x,u) = df/dX
 * --> Take partial derivatives of all equations in the motion model w.r.t. all states in X
 * dx/dx, dx/dy, dx/dh, dx/dv = [1, 0, -dt*sin(h)*v_in, dt*cos(h)] 
 * dy/dx, dy/dy, dy/dh, dy/dv = [0, 1, dt*cos(h)*v_in, dt*sin(h)]
 * dh/dx, dh/dy, dh/dh, dh/dv = [0 0 1 0]
 * dv/dx, dv/dy, dv/dh, dv/dv = [0 0 0 1]
 * 
 * Jacobian of Measurement (observation) Model: H(x) = dh/dX
 * dx_meas/dx, dx_meas/dy, dx_meas/dh, dx_meas/dv = [1, 0, 0, 0]
 * dy_meas/dx, dy_meas/dy, dy_meas/dh, dy_meas/dv = [0, 1, 0, 0]
 * 
 * 
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "KalmanTuner.h"
#include "ekf.hpp"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

// Motion model of the 2D robot, which takes velocity and angular rate as input. States are 2D position, yaw and velocity
Eigen::Vector4d motion_model(const Eigen::Vector4d &X, const Eigen::Vector2d &U, const float &dt)
{
    Eigen::Matrix4d             A;
    Eigen::Matrix<double, 4, 2> B;
    double                      h = X(2);

    A << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    B << dt * std::cos(h), 0.0, dt * std::sin(h), 0.0, 0.0, dt, 1.0, 0.0;

    return (A * X + B * U);
}

// Observation model for the 2D robot. Here, we're emulating GPS measurements.
Eigen::Vector2d observation_model(const Eigen::Vector4d &X)
{
    Eigen::Matrix<double, 2, 4> C;

    C << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;

    return (C * X);
}

Eigen::Vector2d process_input_func(const float &dt)
{
    static float t_total = 0.0;
    double       v       = 1.0;                                   // [m/s]
    double       w       = 0.1 * ((t_total > 20.0) ? 1.0 : -1.0); // [rad/s]
    t_total += dt;
    return Eigen::Vector2d(v, w);
}

// Jacobian of the motion model
Eigen::Matrix4d jacobian_F(const Eigen::Vector4d &X_hat_prio, const Eigen::Vector2d &U, const float &dt)
{
    Eigen::Matrix4d F;
    double          v_in = U(0);
    double          h    = X_hat_prio(2);

    F << 1.0, 0.0, -dt * v_in * std::sin(h), dt * std::cos(h), 0.0, 1.0, dt * v_in * std::cos(h), dt * std::sin(h), 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

    return F;
}

// Jacobian of the measurement model
Eigen::Matrix<double, 2, 4> jacobian_H(const Eigen::Vector4d &X_hat_prio)
{
    // normally jacobian is a function of current aprioristate estimate X,
    // but it happened to be a constant matrix here.
    std::ignore = X_hat_prio;

    Eigen::Matrix<double, 2, 4> H;

    H << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;

    return H;
}

// Calculate the root mean squared error of estimated states and plot
void calculate_RMSE(const std::vector<Eigen::Vector<double, STATESIZE>> &X_true_vec,
                    const std::vector<Eigen::Vector<double, STATESIZE>> &X_est_vec)
{
    double rmse(0);

    for (size_t i = 0; i < X_true_vec.size(); i++)
    {
        rmse += (X_true_vec.at(i)(0) - X_est_vec.at(i)(0)) * (X_true_vec.at(i)(0) - X_est_vec.at(i)(0)) +
                (X_true_vec.at(i)(1) - X_est_vec.at(i)(1)) * (X_true_vec.at(i)(1) - X_est_vec.at(i)(1));
    }
    rmse = rmse / X_true_vec.size();
    std::cout << "RMSE:\n" << rmse << std::endl;
}

int main()
{
    // normal distribution noise for the inputs v,w, and the measurement x,y
    std::default_random_engine       rand_gen(0); // with a seed 0 for repeatibility
    std::normal_distribution<double> normal_dist_noise(0.0, 1.0);
    // scale the noise
    Eigen::Matrix2d meas_noise =
        Eigen::Vector2d(0.5, 0.5).array().square().matrix().asDiagonal(); // about 0.5m measurement noise
    Eigen::Matrix2d input_noise = Eigen::Vector2d(1.0, 30.0 / 180.0 * M_PI)
                                      .array()
                                      .square()
                                      .matrix()
                                      .asDiagonal(); // about 1.0m/s and 30deg. input noise

    // True initial state variable
    Eigen::Vector4d X_true(0, 0, 0, 0);

    std::vector<Eigen::Vector4d> trueStatesVec;
    std::vector<Eigen::Vector2d> trueMeasVec;
    std::vector<Eigen::Vector2d> trueCtrlVec;

    std::vector<Eigen::Vector2d> noisyMeasVec;
    std::vector<Eigen::Vector2d> noisyCtrlVec;

    // We dont take the initial state to optimization since its known
    // Generate dataset
    for (int i = 1; i < 501; i++)
    {
        // Calculate the true states of the system
        Eigen::Vector2d u = process_input_func(DT);
        X_true            = motion_model(X_true, u, DT);
        Eigen::Vector2d z = observation_model(X_true);

        trueCtrlVec.push_back(u);
        trueStatesVec.push_back(X_true);
        trueMeasVec.push_back(z);

        // Add noise to the measurements
        Eigen::Vector2d z_meas =
            z + meas_noise * Eigen::Vector2d(normal_dist_noise(rand_gen), normal_dist_noise(rand_gen));
        // add noise to input
        Eigen::Vector2d u_meas =
            u + input_noise * Eigen::Vector2d(normal_dist_noise(rand_gen), normal_dist_noise(rand_gen));

        noisyCtrlVec.push_back(u_meas);
        noisyMeasVec.push_back(z_meas);
    }

    KalmanTuner kalmanTuner;
    kalmanTuner.Tune(noisyCtrlVec, noisyMeasVec, trueMeasVec);

    ////////////////////////// Now try the tuned parameters

    // Covariance matrices for the Kalman Filter
    Eigen::Matrix2d R  = Eigen::Matrix2d::Zero(); // measurement update covariance
    Eigen::Matrix4d Q  = Eigen::Matrix4d::Zero(); // Prediction update covariance
    Eigen::Matrix4d P0 = Eigen::MatrixXd::Identity(4, 4);
    // Populate the noise matrices with tuned vars
    R(0, 0) = kalmanTuner.filterNoiseParams.at(0);
    R(1, 1) = kalmanTuner.filterNoiseParams.at(1);
    Q(0, 0) = kalmanTuner.filterNoiseParams.at(2);
    Q(1, 1) = kalmanTuner.filterNoiseParams.at(3);
    Q(2, 2) = kalmanTuner.filterNoiseParams.at(4);
    Q(3, 3) = kalmanTuner.filterNoiseParams.at(5);
    // R(0, 0) = 1.0;
    // R(1, 1) = 1.0;
    // Q(0, 0) = 0.1;
    // Q(1, 1) = 0.1;
    // Q(2, 2) = 1.0 / 180.0 * M_PI;
    // Q(3, 3) = 1.0;

    EKF ekf_filter(P0, Q, R, DT);

    // Assign user defined models to the filter
    ekf_filter.motion_model      = motion_model;
    ekf_filter.observation_model = observation_model;
    // Assign user defined Jacobians to the filter
    ekf_filter.jacobian_F = jacobian_F;
    ekf_filter.jacobian_H = jacobian_H;

    Eigen::Vector4d x0(0, 0, 0, 0);
    ekf_filter.Init(x0);

    std::vector<double>                           x_hat_buffer, y_hat_buffer, x_true_buffer, y_true_buffer;
    std::vector<Eigen::Vector<double, STATESIZE>> estStatesVec;
    for (int i = 1; i < 501; i++)
    {
        Eigen::Vector<double, CTRLSIZE> noisyCtrlInput = noisyCtrlVec.at(i - 1);
        Eigen::Vector<double, MEASSIZE> noisyMeas      = noisyMeasVec.at(i - 1);
        // Eigen::Vector<double, MEASSIZE>  trueMeas       = trueMeasVec.at(i - 1);
        Eigen::Vector<double, STATESIZE> trueState = trueStatesVec.at(i - 1);

        ekf_filter.Predict(noisyCtrlInput);
        ekf_filter.Correct(noisyMeas);

        x_hat_buffer.push_back(ekf_filter.GetState()(0));
        y_hat_buffer.push_back(ekf_filter.GetState()(1));
        x_true_buffer.push_back(trueState(0));
        y_true_buffer.push_back(trueState(1));
        estStatesVec.push_back(ekf_filter.GetState());
    }

    calculate_RMSE(trueStatesVec, estStatesVec);

    // Write to file
    std::ofstream file("out.txt");
    for (size_t idx = 0; idx < x_hat_buffer.size(); idx++)
    {
        file << x_hat_buffer.at(idx) << " " << y_hat_buffer.at(idx) << " " << x_true_buffer.at(idx) << " "
             << y_true_buffer.at(idx) << " " << noisyMeasVec.at(idx)(0) << " " << noisyMeasVec.at(idx)(1) << std::endl;
    }
}