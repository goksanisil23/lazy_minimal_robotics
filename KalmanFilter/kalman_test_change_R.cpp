#include <iostream>
#include <vector>
#include <random>
#include <memory>

#include "kalman.hpp"

#define WITHOUT_NUMPY
#include "matplotlibcpp.h"

int main() {

    // The real state value that we're trying to estimate
    double x_real = -0.37727;
    // To generate noisy measurements
    double meas_noise_real_std_dev = 0.1;
    std::default_random_engine meas_gen(0); // with a seed 0 for repeatibility
    std::normal_distribution<double> meas_dist(x_real, meas_noise_real_std_dev);    

    int n = 1; // number of states
    int m = 1; // number of measured variables
    int c = 1; // number of control input variables

    Eigen::MatrixXd A(n,n); // system dynamics matrix
    Eigen::MatrixXd B(n,c); // Input control matrix
    Eigen::MatrixXd C(m,n); // Output matrix
    Eigen::MatrixXd Q(n,n); // Process noise covariance
    Eigen::MatrixXd R(m,m); // Measurement noise covariance
    Eigen::MatrixXd P0(n,n); // Initial state estimate error covariance

    // Trying to estimate the variable that we're directly measuring with some noise
    A << 1;
    B << 1; // we wont provide any input but still need to have a non-zero dim. matrix
    C << 1;

    // Covariance matrices
    Q << 0.00001;
    P0 << 1.0;
    R << (meas_noise_real_std_dev) * (meas_noise_real_std_dev); // take "true" measurement error variance

    // Create the Kalman Filters with varying measurement noise covariances
    KalmanFilter kalman_f(A,B,C,Q,R,P0);
    KalmanFilter kalman_f2(A,B,C,Q,R * std::pow(0.1,2),P0);
    KalmanFilter kalman_f3(A,B,C,Q,R * std::pow(10,2),P0);

    // Some initial guess for the state we're trying to estimate
    Eigen::VectorXd x0(n);
    x0 << 0.0;
    kalman_f.init(x0);
    kalman_f2.init(x0);
    kalman_f3.init(x0);

    // Measurement & control input(if any)
    Eigen::VectorXd y(m);
    Eigen::VectorXd u(c);

    std::vector<double> y_buffer;
    std::vector<double> x_buffer;
    std::vector<double> x_hat_buffer, x2_hat_buffer, x3_hat_buffer;
    std::vector<double> P_buffer, P2_buffer, P3_buffer;
    matplotlibcpp::figure_size(600,400);

    for(int i = 0; i < 250; i++) {
        y << meas_dist(meas_gen); // generate noisy measurement
        u << 0; // No control input in this case
        kalman_f.prediction_update(u);
        kalman_f.innovation_update(y);
        kalman_f2.prediction_update(u);
        kalman_f2.innovation_update(y);
        kalman_f3.prediction_update(u);
        kalman_f3.innovation_update(y);                


        x_hat_buffer.push_back(kalman_f.get_state()[0]);
        x2_hat_buffer.push_back(kalman_f2.get_state()[0]);
        x3_hat_buffer.push_back(kalman_f3.get_state()[0]);
        x_buffer.push_back(x_real);
        y_buffer.push_back(y[0]);
        P_buffer.push_back(kalman_f.get_P()[0]);
        P2_buffer.push_back(kalman_f2.get_P()[0]);
        P3_buffer.push_back(kalman_f3.get_P()[0]);

        // // Rendering
        // matplotlibcpp::clf();
        // matplotlibcpp::subplot(1, 2, 1);
        // matplotlibcpp::named_plot("x hat", x_hat_buffer);
        // matplotlibcpp::named_plot("measurement", y_buffer, ".r");
        // matplotlibcpp::named_plot("x true", x_buffer, "y");
        // matplotlibcpp::grid(true);
        // matplotlibcpp::legend();
        // matplotlibcpp::subplot(1, 2, 2);
        // matplotlibcpp::named_plot("P", P_buffer);
        // matplotlibcpp::grid(true);
        // matplotlibcpp::pause(0.0001);        
    }
        // Rendering
        matplotlibcpp::clf();
        // matplotlibcpp::subplot(1, 2, 1);
        matplotlibcpp::named_plot("x hat: Exact R", x_hat_buffer);
        matplotlibcpp::named_plot("x2 hat: Small R", x2_hat_buffer);
        matplotlibcpp::named_plot("x3 hat: High R", x3_hat_buffer);
        matplotlibcpp::named_plot("measurement", y_buffer, ".c");
        matplotlibcpp::named_plot("x true", x_buffer, "--k");
        matplotlibcpp::grid(true);
        matplotlibcpp::legend();
        matplotlibcpp::ylabel("State variable unit");
        matplotlibcpp::xlabel("iterations");
        // matplotlibcpp::subplot(1, 2, 2);
        // matplotlibcpp::named_plot("P: Exact R", P_buffer);
        // matplotlibcpp::named_plot("P2: Exact R", P2_buffer);
        // matplotlibcpp::named_plot("P3: Exact R", P3_buffer);
        // matplotlibcpp::grid(true);
        // matplotlibcpp::legend();
        // matplotlibcpp::ylabel("State est. err. covariance");
        // matplotlibcpp::xlabel("iterations");        
        matplotlibcpp::pause(0);   



    return 0;
}