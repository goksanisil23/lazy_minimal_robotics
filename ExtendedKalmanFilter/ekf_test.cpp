/**
 * @brief This is a simple example usage of the Extended Kalman Filter
 * It is used to estimate the 2D position(x,y) and heading(h) of a mobile robot, 
 * by using noisy linear velocity(v) and angular rate(w) inputs, 
 * as well as noisy (x,y) position measurements.
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

#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "EKF.hpp"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

/* 
    User defined EKF models
*/

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

Eigen::Vector2d process_input_func(const float &dt)
{
    static float t_total = 0.0;
    double       v       = 1.0;                                   // [m/s]
    double       w       = 0.1 * ((t_total > 20.0) ? 1.0 : -1.0); // [rad/s]
    t_total += dt;
    return Eigen::Vector2d(v, w);
}

/* ----------------- End of user defined EKF models ---------------------------- */

// Calculate the root mean squared error of estimated states and plot
void calculate_RMSE(const std::vector<Eigen::VectorXd> &X_true_buffer, const std::vector<Eigen::VectorXd> &X_est_buffer)
{
    std::vector<double> err_x, err_y, err_h, err_v;
    Eigen::Vector4d     rmse(0, 0, 0, 0);

    for (size_t i = 0; i < X_true_buffer.size(); i++)
    {
        // calculate sample & state based absolute error
        err_x.push_back((std::fabs(X_true_buffer.at(i)(0) - X_est_buffer.at(i)(0))));
        err_y.push_back((std::fabs(X_true_buffer.at(i)(1) - X_est_buffer.at(i)(1))));
        err_h.push_back((std::fabs(X_true_buffer.at(i)(2) - X_est_buffer.at(i)(2))));
        err_v.push_back((std::fabs(X_true_buffer.at(i)(3) - X_est_buffer.at(i)(3))));
        // accumulate squared error
        Eigen::VectorXd residual = X_true_buffer.at(i) - X_est_buffer.at(i);
        residual                 = residual.array() * residual.array();
        rmse += residual;
    }
    rmse = rmse / X_true_buffer.size();
    rmse = rmse.array().sqrt();
    std::cout << "RMSE:\n" << rmse << std::endl;
}

int main()
{

    float dt = 0.1; // simulation step time

    // Covariance matrices for the Kalman Filter
    Eigen::Vector4d q_variances(0.1, 0.1, 1.0 / 180.0 * M_PI, 1.0); // variances of x,y,yaw,velocity in prediction
    Eigen::Matrix4d Q = q_variances.array().square().matrix().asDiagonal();  // Prediction update covariance
    Eigen::Vector2d r_variances(1.0, 1.0);                                   // variances of x,y in observation
    Eigen::Matrix2d R  = r_variances.array().square().matrix().asDiagonal(); // measurement update covariance
    Eigen::Matrix4d P0 = Eigen::MatrixXd::Identity(4, 4);

    EKF ekf_filter(P0, Q, R, dt);

    // Assign user defined models to the filter
    ekf_filter.motion_model      = motion_model;
    ekf_filter.observation_model = observation_model;
    // Assign user defined Jacobians to the filter
    ekf_filter.jacobian_F = jacobian_F;
    ekf_filter.jacobian_H = jacobian_H;

    // Some initial guess for the states we're trying to estimate [x y h v]
    Eigen::Vector4d x0(0, 0, 0, 0);
    ekf_filter.Init(x0);

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
    // Dead reckoning
    Eigen::Vector4d X_dead_reckon(0, 0, 0, 0);

    std::vector<double>          x_hat_buffer{x0(0)}, y_hat_buffer{x0(1)};
    std::vector<double>          x_true_buffer{X_true(0)}, y_true_buffer{X_true(1)};
    std::vector<double>          x_dead_buffer{X_dead_reckon(0)}, y_dead_buffer{X_dead_reckon(1)};
    std::vector<double>          z_meas_buffer_x, z_meas_buffer_y;
    std::vector<double>          P_buffer_x, P_buffer_y;
    std::vector<double>          nis_buffer;
    std::vector<Eigen::VectorXd> X_true_buffer;
    std::vector<Eigen::VectorXd> X_est_buffer;

    std::cout << "press key to continue...\n";
    std::cin.get();

    for (int i = 0; i < 500; i++)
    {
        // Calculate the true states of the system
        Eigen::Vector2d u = process_input_func(dt);
        X_true            = motion_model(X_true, u, dt);
        Eigen::Vector2d z = observation_model(X_true);

        // Add noise to the measurements
        Eigen::Vector2d z_meas =
            z + meas_noise * Eigen::Vector2d(normal_dist_noise(rand_gen),
                                             normal_dist_noise(rand_gen)); // add noise to measurement
        Eigen::Vector2d u_meas = u + input_noise * Eigen::Vector2d(normal_dist_noise(rand_gen),
                                                                   normal_dist_noise(rand_gen)); // add noise to input

        // Calculate dead-reckoning --> just iterate the motion model with measurement (no correction)
        X_dead_reckon = motion_model(X_dead_reckon, u_meas, dt);

        // Execute EKF with the noisy measurements
        ekf_filter.Predict(u_meas);
        ekf_filter.Correct(z_meas);

        // store the history
        x_hat_buffer.push_back(ekf_filter.GetState()(0));
        y_hat_buffer.push_back(ekf_filter.GetState()(1));
        x_true_buffer.push_back(X_true(0));
        y_true_buffer.push_back(X_true(1));
        x_dead_buffer.push_back(X_dead_reckon(0));
        y_dead_buffer.push_back(X_dead_reckon(1));
        z_meas_buffer_x.push_back(z_meas(0));
        z_meas_buffer_y.push_back(z_meas(1));
        P_buffer_x.push_back(ekf_filter.GetP()(0, 0));
        P_buffer_y.push_back(ekf_filter.GetP()(1, 1));
        nis_buffer.push_back(ekf_filter.GetNis());
        X_est_buffer.push_back(ekf_filter.GetState());
        X_true_buffer.push_back(X_true);

        // plt::clf();
        // plt::named_plot("x hat", x_hat_buffer, y_hat_buffer);
        // plt::named_plot("x true", x_true_buffer, y_true_buffer, "y");
        // plt::named_plot("x deadreckon", x_dead_buffer, y_dead_buffer, "k");
        // plt::named_plot("measurement", z_meas_buffer_x, z_meas_buffer_y, ".g");
        // plt::grid(true);
        // plt::legend();
        // plt::ylabel("y [m.]");
        // plt::xlabel("x [m.]");
        // plt::pause(0.001);
    }

    plt::figure(); // declare a new figure (optional if only one is used)
    std::vector<int> a{1, 2, 3, 4};
    plt::plot(a); // automatic coloring: tab:blue
    plt::show();

    // double const chi_table_dof2_0_05 = 5.991;
    // std::vector<double> chi_buffer(nis_buffer.size(), chi_table_dof2_0_05);
    // plt::clf();
    // plt::named_plot("NIS", nis_buffer, ".-");
    // plt::named_plot("chi val", chi_buffer);
    // plt::grid(true);
    // plt::legend();
    // plt::ylabel("y [m.]");
    // plt::xlabel("x [m.]");
    // plt::pause(0.00);

    // Plotting
    // matplot::hold(matplot::on);
    // matplot::plot(y_true_buffer, "o")->marker_size(MARKER_SIZE).line_width(LINE_WIDTH).display_name("ground truth");
    // matplot::legend()->font_size(MARKER_SIZE);
    // matplot::grid(matplot::on);
    // matplot::show();

    // calculate_RMSE(X_true_buffer, X_est_buffer);

    return 0;
}