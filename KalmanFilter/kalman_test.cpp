/**
 * @brief This is a simple example usage of the vanilla Kalman Filter
 * It is used to estimate the value of a constant variable, through noisy measurements (y)
 *
 * x_(k+1) = 1 * x_(k) + 1 * u_(k) + w_(k) --> We do not have any input to the system, so u_(k) = 0 
 * y_(k) = 1 * x_(k) + v_(k) --> Measurements are directly the state we want to estimate, though noisy
 */

#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "kalman.hpp"

#include "matplotlibcpp.h"

int main()
{

    // The real state value that we're trying to estimate
    double x_real = -0.37727;
    // To generate noisy measurements
    double                           meas_noise_real_std_dev = 0.1;
    std::default_random_engine       meas_gen(0); // with a seed 0 for repeatibility
    std::normal_distribution<double> meas_dist(x_real, meas_noise_real_std_dev);

    int n = 1; // number of states
    int m = 1; // number of measured variables
    int c = 1; // number of control input variables

    Eigen::MatrixXd A(n, n);  // system dynamics matrix
    Eigen::MatrixXd B(n, c);  // Input control matrix
    Eigen::MatrixXd C(m, n);  // Output matrix
    Eigen::MatrixXd Q(n, n);  // Process noise covariance
    Eigen::MatrixXd R(m, m);  // Measurement noise covariance
    Eigen::MatrixXd P0(n, n); // Initial state estimate error covariance

    // Trying to estimate the variable that we're directly measuring with some noise
    A << 1;
    B << 1; // we wont provide any input but still need to have a non-zero dim. matrix
    C << 1;

    // Covariance matrices for the Kalman Filter
    Q << 0.00001;
    P0 << 1.0;
    R << (meas_noise_real_std_dev) * (meas_noise_real_std_dev); // take "true" measurement error variance

    // Create the Kalman Filter
    KalmanFilter kalman_f(A, B, C, Q, R, P0);

    // Some initial guess for the state we're trying to estimate
    Eigen::VectorXd x0(n);
    x0 << 0.0;
    kalman_f.init(x0);

    // Measurement & control input(if any)
    Eigen::VectorXd y(m);
    Eigen::VectorXd u(c);

    std::vector<double> y_buffer;
    std::vector<double> x_buffer;
    std::vector<double> x_hat_buffer;
    std::vector<double> P_buffer;
    matplotlibcpp::figure_size(600, 400);

    for (int i = 0; i < 200; i++)
    {
        y << meas_dist(meas_gen); // generate noisy measurement
        u << 0;                   // No control input in this case
        kalman_f.prediction_update(u);
        kalman_f.innovation_update(y);

        // store the history
        x_hat_buffer.push_back(kalman_f.get_state()(0));
        x_buffer.push_back(x_real);
        y_buffer.push_back(y(0));
        P_buffer.push_back(kalman_f.get_P()(0));
    }
    // Rendering
    matplotlibcpp::clf();
    matplotlibcpp::subplot(1, 2, 1);
    matplotlibcpp::named_plot("x hat", x_hat_buffer);
    matplotlibcpp::named_plot("measurement", y_buffer, ".r");
    matplotlibcpp::named_plot("x true", x_buffer, "y");
    matplotlibcpp::grid(true);
    matplotlibcpp::legend();
    matplotlibcpp::ylabel("State variable unit");
    matplotlibcpp::xlabel("iterations");
    matplotlibcpp::subplot(1, 2, 2);
    matplotlibcpp::named_plot("P", P_buffer);
    matplotlibcpp::grid(true);
    matplotlibcpp::legend();
    matplotlibcpp::ylabel("State est. err. covariance");
    matplotlibcpp::xlabel("iterations");
    matplotlibcpp::pause(0);

    return 0;
}