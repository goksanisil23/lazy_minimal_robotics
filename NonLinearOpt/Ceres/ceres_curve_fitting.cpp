// This is a simple example usage of the Ceres library, for a curve fitting
// problem through minimizing a non-linear least squares cost function.
// We're trying to estimate the parameters in y = exp(m*x + c) through noisy
// observations

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/iteration_callback.h>
#include <ceres/types.h>
#include <chrono>
#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <vector>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

constexpr int N_SAMPLES = 100;
constexpr double W_SIGMA = 1.0; // Observation noise
constexpr std::pair<double, double> SAMPLE_INTERVAL{0.0, 5.0};
constexpr double SAMPLE_SPACING =
    (SAMPLE_INTERVAL.second - SAMPLE_INTERVAL.first) / (N_SAMPLES - 1);

// Ground truth equation parameters
constexpr double M = 0.3;
constexpr double C = 0.1;

// Define the residual
struct ExponentialResidual {
  ExponentialResidual(double x, double y) : x_(x), y_(y) {}

  template <typename T>
  bool operator()(const T *const m, const T *c, T *residual) const {
    residual[0] = y_ - exp(m[0] * x_ + c[0]);
    return true;
  }

private:
  const double x_;
  const double y_;
};

// Define the callback function that'll be thrown every step of iteration
// which is used to monitor the current estimation values of the states
// We send the optimized variables as a reference to this callback class
class MyIterCallback : public ceres::IterationCallback {
public:
  MyIterCallback(const double &m_est, const double &c_est)
      : m_est_(m_est), c_est_(c_est) {}

  ceres::CallbackReturnType
  operator()(const ceres::IterationSummary &summary) override {
    std::cout << "Iter: " << summary.iteration << " m: " << m_est_
              << " c:" << c_est_ << std::endl;

    return ceres::SOLVER_CONTINUE;
  }

private:
  // Reference to the current estimated optimization variable
  const double &m_est_;
  const double &c_est_;
};

// The real function we're trying to approximate
double evaluateObservation(const double &x) {
  static cv::RNG rand_gen; // random number generator

  double noise = rand_gen.gaussian(W_SIGMA * W_SIGMA);
  double gt_sample = cv::exp(M * x + C) + noise;
  return gt_sample;
}

// Optimized model sampled at the observation points
void evaluateModel(const double &m_est, const double &c_est,
                   const std::vector<double> &x_gt,
                   std::vector<double> &y_est) {
  int idx = 0;
  for (double x = SAMPLE_INTERVAL.first; x <= SAMPLE_INTERVAL.second;
       x += SAMPLE_SPACING) {
    y_est.at(idx) = exp(m_est * x + c_est);
    idx++;
  }
}

int main() {
  // Initial guesses for the optimization parameters
  double m = 0.0;
  double c = 0.0;

  // Get all the observation samples from the osbervation equation
  int idx = 0;
  std::vector<double> x_gt(N_SAMPLES), y_gt(N_SAMPLES);
  for (double x = SAMPLE_INTERVAL.first; x <= SAMPLE_INTERVAL.second;
       x += SAMPLE_SPACING) {
    x_gt.at(idx) = x;
    y_gt.at(idx) = evaluateObservation(x);

    idx++;
  }

  // Construct the Ceres optimization problem
  // by adding residual blocks constructed with observation samples
  ceres::Problem ceres_problem;
  for (int i = 0; i < N_SAMPLES; i++) {
    ceres_problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
            new ExponentialResidual(x_gt.at(i), y_gt.at(i))),
        nullptr, &m, &c);
  }

  // Define solver options
  ceres::Solver::Options solver_options;
  solver_options.max_num_iterations = 25;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.minimizer_progress_to_stdout = true;
  solver_options.update_state_every_iteration = true;
  MyIterCallback my_callback(m, c);
  solver_options.callbacks.push_back(&my_callback);

  // Solve and summarize
  ceres::Solver::Summary solver_summary;
  ceres::Solve(solver_options, &ceres_problem, &solver_summary);
  std::cout << solver_summary.BriefReport() << "\n";
  std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
  std::cout << "Final   m: " << m << " c: " << c << "\n";

  // Show the fit model
  std::vector<double> y_est(x_gt.size());
  evaluateModel(m, c, x_gt, y_est);
  plt::figure_size(800, 600);
  // plt::clf();
  plt::plot(x_gt, y_gt, ".");
  // plt::named_plot("y true", x_gt, y_gt, ".");
  // plt::named_plot("y est", x_gt, y_est, "y");
  plt::grid(true);
  // plt::legend();
  plt::ylabel("y ");
  plt::xlabel("x ");
  plt::pause(1);

  return 0;
}