// This is a sample example of Gauss-Newton method, where we're trying to
// estimate the non-linear function y = exp(a*x^2 + b*x + c) + w, where w is the
// Gaussian measurement noise.

#include <chrono>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <opencv4/opencv2/core.hpp>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

// Container for the sample points
struct Sample
{
    double x, y;
};

constexpr int                       N_PARAM = 3; // number of parameters of our model
constexpr std::pair<double, double> SAMPLE_INTERVAL{0.0, 1.0};
constexpr int                       N_SAMPLES      = 100;
constexpr int                       N_MAX_ITER     = 100;
constexpr double                    W_SIGMA        = 1.0; // Observation noise
constexpr double                    INV_SIGMA      = 1.0 / W_SIGMA;
constexpr double                    SAMPLE_SPACING = (SAMPLE_INTERVAL.second - SAMPLE_INTERVAL.first) / (N_SAMPLES - 1);
constexpr double                    RMSE_THRESHOLD = 0.0001;

// Ground truth equation parameters
constexpr double A = 1.0;
constexpr double B = 2.0;
constexpr double C = 1.0;

double          P[N_PARAM]; // current guess for the model parameters
Sample          samples[N_SAMPLES];
Eigen::MatrixXd residual(N_SAMPLES,
                         1); // container where we store the error btw GT and model
Eigen::MatrixXd J(N_SAMPLES,
                  N_PARAM); // Jacobian but concatanated at each sample

///////////////////////////////////

// The real function we're trying to approximate
double evaluateObservation(const double &x)
{
    static cv::RNG rand_gen; // random number generator

    double noise     = rand_gen.gaussian(W_SIGMA * W_SIGMA);
    double gt_sample = cv::exp(A * x * x + B * x + C) + noise;
    return gt_sample;
}

// User defined model that we're trying to bring closer to GT
double evaluateModel(const double &x)
{
    double model_estimate = cv::exp(P[0] * x * x + P[1] * x + P[2]);
    return model_estimate;
}

// Jacobian of the model function
Eigen::VectorXd calculateJacobian(const double &x)
{
    Eigen::VectorXd j(N_PARAM);
    j(0) = x * x * cv::exp(P[0] * x * x + P[1] * x + P[2]); // de / dP(1)
    j(1) = x * cv::exp(P[0] * x * x + P[1] * x + P[2]);     // de / dP(2)
    j(2) = cv::exp(P[0] * x * x + P[1] * x + P[2]);         // de / dP(2)

    return j;
}

int main()
{
    // Initial parameter estimates
    P[0] = 2.0;
    P[1] = -1.0;
    P[2] = 5.0;

    // Get all the observation samples from the osbervation equation
    int                 idx = 0;
    std::vector<double> x_gt(N_SAMPLES), y_gt(N_SAMPLES);
    for (double x = SAMPLE_INTERVAL.first; x <= SAMPLE_INTERVAL.second; x += SAMPLE_SPACING)
    {
        Sample sample;
        sample.x     = x;
        sample.y     = evaluateObservation(x);
        samples[idx] = sample;

        x_gt.at(idx) = sample.x;
        y_gt.at(idx) = sample.y;

        idx++;
    }

    // Visualization variables
    matplotlibcpp::figure_size(800, 600);
    std::cin.get();

    // Run Gauss-Newton
    for (int iter = 0; iter < N_MAX_ITER; iter++)
    {
        std::vector<double> y_est(N_SAMPLES);
        double              rmse = 0.0;
        for (int i_sample = 0; i_sample < N_SAMPLES; i_sample++)
        {
            y_est.at(i_sample)    = evaluateModel(samples[i_sample].x);
            residual(i_sample, 0) = (samples[i_sample].y - y_est.at(i_sample));
            rmse += residual(i_sample, 0) * residual(i_sample, 0);
            Eigen::VectorXd jacobian = calculateJacobian(samples[i_sample].x);
            for (int i_param = 0; i_param < N_PARAM; i_param++)
            {
                J(i_sample, i_param) = jacobian(i_param);
            }
        }
        rmse = sqrt(rmse / double(N_SAMPLES));
        std::cout << "RMSE: " << rmse << std::endl;

        plt::clf();
        plt::named_plot("y true", x_gt, y_gt, ".");
        plt::named_plot("y est", x_gt, y_est, "y");
        plt::grid(true);
        plt::legend();
        plt::ylabel("y ");
        plt::xlabel("x ");
        plt::pause(1);

        if (rmse < RMSE_THRESHOLD)
        {
            std::cout << "converged in " << iter << " iterations" << std::endl;
            break;
        }

        // Here we solve the Gauss-Newton
        Eigen::MatrixXd JTJ = J.transpose() * J;
        // Remember that our linear equation to solve is
        // (J^T * J) * h = J^T * (y - y_hat)
        Eigen::VectorXd h = (JTJ).colPivHouseholderQr().solve(J.transpose() * residual);
        for (int i_param = 0; i_param < N_PARAM; i_param++)
        {
            P[i_param] += h(i_param);
        }
    }
}