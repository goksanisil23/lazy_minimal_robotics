// This is a simple example usage of g2o library, for a curve fitting problem
// through formulating the non-linear least squares cost function as a graph
// where the edges of the graph are the noisy observations we make along the
// true curve and the single node of the graph is the parameters of the curve we
// want to find out.

#include "g2o/core/auto_differentiation.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/sparse_optimizer.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/src/Core/Matrix.h>

#include <iostream>
#include <matplot/matplot.h>
#include <opencv4/opencv2/core/core.hpp>
#include <string>
#include <vector>

// ************  Constants of this example ************ //
constexpr int N_SAMPLES = 100;
constexpr double W_SIGMA = 0.5; // Observation noise
constexpr std::pair<double, double> SAMPLE_INTERVAL{0.0, 5.0};
constexpr double SAMPLE_SPACING =
    (SAMPLE_INTERVAL.second - SAMPLE_INTERVAL.first) / (N_SAMPLES - 1);

constexpr int N_ITER = 10;
// Ground truth equation parameters
constexpr double M = 0.3;
constexpr double C = 0.1;
// ************ ************ ************ //

// Define and evaluate the model function whose parameters we're trying to
// optimize
template <typename T>
T evaluteModelAtSamplePoint(const T &m_est, const T &c_est,
                            const double &x_gt) {
  return exp(m_est * x_gt + c_est);
}

// The real function we're trying to approximate
void getObservation(const double &x, double &gt_sample,
                    double &observation_sample, const int &idx) {
  static cv::RNG rand_gen; // random number generator
  double noise;
  if (idx % 10 == 0) {
    noise = 2.5; // to create some outliers
  } else {
    noise = rand_gen.gaussian(W_SIGMA * W_SIGMA);
  }
  gt_sample = cv::exp(M * x + C);
  //   observation_sample = gt_sample;
  observation_sample = gt_sample + noise;
}

// Current model sampled at all the observation points
void evaluateModel(const double &m_est, const double &c_est,
                   const std::vector<double> &x_gt,
                   std::vector<double> &y_est) {
  int idx = 0;
  for (double x = SAMPLE_INTERVAL.first; x <= SAMPLE_INTERVAL.second;
       x += SAMPLE_SPACING) {
    y_est.at(idx) = evaluteModelAtSamplePoint(m_est, c_est, x);
    idx++;
  }
}

// ************ g2o Elements ************ //

// Vertex parameters: Dimension of the vertex (how many variables in the
// vertex), internal type to represent it
class CurveFittingVertex : public g2o::BaseVertex<2, Eigen::Vector2d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void oplusImpl(const double *update) override {
    _estimate += Eigen::Vector2d(update);
  }

  // These need to be overwritten o/w compile error by g2o
  virtual void setToOriginImpl() {}
  virtual bool read(std::istream &in) {}
  virtual bool write(std::ostream &out) const {}
};

// Error model template parameters: observation
// dimension, type, connecting vertex type
class CurveFittingEdge
    : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
  CurveFittingEdge(double x_sample) : _x_sample(x_sample) {}

  // Templated to utilize auto-diff
  template <typename T> bool operator()(const T *params, T *residual) const {
    residual[0] = T(_measurement) -
                  evaluteModelAtSamplePoint(params[0], params[1], _x_sample);
    return true;
  }

  virtual bool read(std::istream &) { return false; }
  virtual bool write(std::ostream &) const { return false; }

private:
  double _x_sample;

  G2O_MAKE_AUTO_AD_FUNCTIONS // use autodiff
};

// ************  MAIN ************ //

int main() {
  // Initial guesses for the optimization parameters
  double m = 0.0;
  double c = 0.0;

  // Get all the noisy observation and ground truth samples from the observation
  // equation
  int idx = 0;
  std::vector<double> x_gt(N_SAMPLES), y_gt(N_SAMPLES), y_obs(N_SAMPLES);
  for (double x = SAMPLE_INTERVAL.first; x <= SAMPLE_INTERVAL.second;
       x += SAMPLE_SPACING) {
    x_gt.at(idx) = x;
    getObservation(x, y_gt.at(idx), y_obs.at(idx), idx);

    idx++;
  }

  // Construct g2o optimization problem
  // We have 2 optimization variables, dimension of residual is 1
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<2, 1>> BlockSolverType;
  // Linear solver with the residual and parameter size defined above
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
      LinearSolverType;

  //   Specify the gradient descent method
  auto g2o_solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer g2o_optimizer;
  g2o_optimizer.setAlgorithm(g2o_solver);
  g2o_optimizer.setVerbose(true);

  // Repeat the experiment twice: With and without Robust Kernel
  std::vector<double> y_est(x_gt.size());
  std::vector<double> y_est_with_kernel(x_gt.size());

  std::vector<bool> robust_kernel_usage{false, true};
  for (bool use_robuts_kernel : robust_kernel_usage) {
    // Add the single vertex to the graph
    CurveFittingVertex *vertex = new CurveFittingVertex();
    vertex->setEstimate(Eigen::Vector2d(m, c));
    vertex->setId(0);
    g2o_optimizer.addVertex(vertex);

    // Add edges (observations) to the graph
    for (int obs_idx = 0; obs_idx < N_SAMPLES; obs_idx++) {
      CurveFittingEdge *edge = new CurveFittingEdge(x_gt.at(obs_idx));
      edge->setId(obs_idx);
      // connect the 0th vertex on the graph to vertex (2nd arg)
      edge->setVertex(0, vertex);
      edge->setMeasurement(y_obs.at(obs_idx));
      edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
      if (use_robuts_kernel)
        edge->setRobustKernel(new g2o::RobustKernelCauchy());
      g2o_optimizer.addEdge(edge);
    }

    g2o_optimizer.initializeOptimization();
    g2o_optimizer.optimize(N_ITER);
    Eigen::Vector2d final_estimate = vertex->estimate();

    // Summary
    std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
    std::cout << "Final   m: " << final_estimate(0)
              << " c: " << final_estimate(1) << "\n";
    if (use_robuts_kernel)
      evaluateModel(final_estimate(0), final_estimate(1), x_gt,
                    y_est_with_kernel);
    else
      evaluateModel(final_estimate(0), final_estimate(1), x_gt, y_est);
    ;

    // Reset the experiment
    g2o_optimizer.clear();
  }

  // Show the fit model
  matplot::plot(x_gt, y_gt, "-")->line_width(3).display_name("ground truth");
  matplot::hold(matplot::on);
  matplot::plot(x_gt, y_obs, "o")->line_width(2).display_name("observation");
  matplot::plot(x_gt, y_est, "-s")->line_width(2).display_name("estimate");
  matplot::plot(x_gt, y_est_with_kernel, "-rx")
      ->line_width(2)
      .display_name("estimate with Cauchy Robust Kernel");
  matplot::legend();
  matplot::grid(matplot::on);
  matplot::show();

  return 0;
}