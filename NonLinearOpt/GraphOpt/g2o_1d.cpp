// This is a simple example usage of g2o library, for a 1D localization problem
// through formulating the non-linear least squares cost function as a graph
// where the edges of the graph are the noisy observations robot makes towards a
// landmark and the vertices are the poses of the robot and the landmark whose initial
// estimates are obtained via some drifty dead-reckoning.

#include "g2o/core/auto_differentiation.h"
#include "g2o/core/base_binary_edge.h"
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
constexpr int N_ITER = 40;
constexpr int N_POSES = 100;
constexpr double ROBOT_STEP = 1.0; // true displacement (m)
constexpr double W_SIGMA = 0.3;    // Observation and deadreckon noise

constexpr int MARKER_SIZE = 20;
constexpr int LINE_WIDTH = 6;
// ************ ************ ************ //

// ************ g2o Elements ************ //

// Vertex parameters: Dimension of the vertex (how many variables in the
// vertex), internal type to represent it
class Vertex1DPose : public g2o::BaseVertex<1, double>
{
public:
    void oplusImpl(const double *update) override
    {
        _estimate += double(update[0]);
    }

    double project(const double &landmarkBelief)
    {
        // distance from landmark to robot (which is also what sensor measures)
        return landmarkBelief - _estimate;
    }

    // These need to be overwritten o/w compile error by g2o
    virtual void setToOriginImpl()
    {
    }
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}
};

class Vertex1DLandmark : public g2o::BaseVertex<1, double>
{
public:
    void oplusImpl(const double *update) override
    {
        _estimate += double(update[0]);
    }

    // These need to be overwritten o/w compile error by g2o
    virtual void setToOriginImpl() {}
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}
};

// Error model template parameters: observation dimension, observation type, connecting vertex type(s)
class OdometryEdgeProjection
    : public g2o::BaseBinaryEdge<1, double, Vertex1DPose, Vertex1DLandmark>
{
public:
    OdometryEdgeProjection(double odom_idx) : _odom_idx(odom_idx) {}

    virtual void computeError() override
    {
        auto v0 = static_cast<Vertex1DPose *>(_vertices[0]);
        auto v1 = static_cast<Vertex1DLandmark *>(_vertices[1]);
        auto projectedLandmark = v0->project(v1->estimate());
        _error[0] = projectedLandmark - _measurement;
    }

    virtual bool read(std::istream &) { return false; }
    virtual bool write(std::ostream &) const { return false; }

private:
    int _odom_idx;
};

// ************  MAIN ************ //

int main()
{
    // Get all the noisy observation and ground truth samples
    double landmark_true{(N_POSES - 1) * ROBOT_STEP / 2.0}; // landmark is at the half-way
    std::vector<double> y_gt(N_POSES), y_odom(N_POSES), landmark_obs(N_POSES);
    cv::RNG rand_gen; // random number generator

    y_gt.at(0) = 0.0;
    y_odom.at(0) = y_gt.at(0);                                                         // we know the exact initial position
    landmark_obs.at(0) = (landmark_true - y_gt.at(0));                                 // true landmark distance
    landmark_obs.at(0) += ((1.0 - rand_gen.gaussian(W_SIGMA * W_SIGMA)) * ROBOT_STEP); // add noise to observation
    double landmark_init_est = y_gt.at(0) + landmark_obs.at(0);                        // we use the landmark observation from 1st pose as initial estimate of the landmark

    for (int i = 1; i < N_POSES; i++)
    {
        y_gt.at(i) = y_gt.at(i - 1) + ROBOT_STEP;
        double odom_step_with_noise = (1.0 + std::fabs(rand_gen.gaussian(W_SIGMA * W_SIGMA))) * ROBOT_STEP; // scale the noise with the unit step of robot
        y_odom.at(i) = y_odom.at(i - 1) + odom_step_with_noise;
        landmark_obs.at(i) = (landmark_true - y_gt.at(i));                         // true landmark distance
        landmark_obs.at(i) += (rand_gen.gaussian(W_SIGMA * W_SIGMA) * ROBOT_STEP); // add noise to observation
    }
    std::cout << "landmark true: " << landmark_true << std::endl;
    std::cout << "landmark init est: " << landmark_init_est << std::endl;

    // Construct g2o optimization problem
    // We have 1 optimization variable per vertex, dimension of residual is 1
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<1, 1>> BlockSolverType;
    // Linear solver with the residual and parameter size defined above
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    //   Specify the gradient descent method
    auto g2o_solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer g2o_optimizer;
    g2o_optimizer.setAlgorithm(g2o_solver);
    g2o_optimizer.setVerbose(true);

    std::vector<Vertex1DPose *> poseVertices;
    std::vector<Vertex1DLandmark *> landmarkVertices;
    // Add the robot pose vertices to the graph
    for (int i = 0; i < y_odom.size(); i++)
    {
        Vertex1DPose *vertex = new Vertex1DPose();
        vertex->setEstimate(y_odom.at(i));
        vertex->setId(i);
        if (i == 0)
            vertex->setFixed(true);
        g2o_optimizer.addVertex(vertex);
        poseVertices.push_back(vertex);
    }
    // Add landmark vertex to the graph
    Vertex1DLandmark *vertex = new Vertex1DLandmark();
    vertex->setEstimate(landmark_init_est); // use the 1st landmark observation projection as initial estimate
    vertex->setId(y_odom.size());
    vertex->setMarginalized(true);
    g2o_optimizer.addVertex(vertex);
    landmarkVertices.push_back(vertex);

    // Add edges (observations) to the graph
    for (int obs_idx = 0; obs_idx < y_odom.size(); obs_idx++)
    {
        OdometryEdgeProjection *edge = new OdometryEdgeProjection(obs_idx);
        edge->setId(obs_idx);
        // connect the 0th vertex of THIS EDGE to a given vertex
        edge->setVertex(0, poseVertices.at(obs_idx));
        // connect the 1st vertex of THIS EDGE to a given vertex
        edge->setVertex(1, landmarkVertices.at(0));

        edge->setMeasurement(landmark_obs.at(obs_idx));
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        // if (use_robuts_kernel)
        //     edge->setRobustKernel(new g2o::RobustKernelCauchy());
        g2o_optimizer.addEdge(edge);
    }

    g2o_optimizer.initializeOptimization();
    g2o_optimizer.optimize(N_ITER);

    std::vector<double> finalPoses;
    double rmse_before{0}, rmse_after{0};
    for (int i = 0; i < poseVertices.size(); i++)
    {
        std::cout << "robot: " << i << " pos: " << poseVertices.at(i)->estimate() << std::endl;
        finalPoses.push_back(poseVertices.at(i)->estimate());
        rmse_before += std::pow(y_gt.at(i) - y_odom.at(i), 2);
        rmse_after += std::pow(y_gt.at(i) - finalPoses.at(i), 2);
    }
    std::cout << "landmark final est: " << landmarkVertices.at(0)->estimate() << std::endl;

    // RMSE before and after
    std::cout << "RMSE before: " << rmse_before / y_gt.size() << std::endl;
    std::cout << "RMSE after: " << rmse_after / y_gt.size() << std::endl;

    // Plotting
    matplot::hold(matplot::on);
    std::vector<double> x_axis(y_gt.size(), 0.0);
    matplot::plot(y_gt, x_axis, "o")->marker_size(MARKER_SIZE).line_width(LINE_WIDTH).display_name("ground truth");
    matplot::plot(y_odom, x_axis, "s")->marker_size(MARKER_SIZE).line_width(LINE_WIDTH).display_name("dead-reckon");
    matplot::plot(finalPoses, x_axis, "x")->marker_size(MARKER_SIZE).line_width(LINE_WIDTH).display_name("estimate");
    matplot::legend()->font_size(MARKER_SIZE);
    matplot::grid(matplot::on);
    matplot::show();

    return 0;
}

// #include <opencv4/opencv2/core/core.hpp>
// #include <iostream>
// #include <matplot/matplot.h>

// int main()
// {
//     static cv::RNG rand_gen; // random number generator
//     constexpr double W_SIGMA = 0.1;

//     matplot::hold(matplot::on);
//     std::vector<double> x_axis(100, 0.0);
//     std::vector<double> y_axis(100, 0.0);

//     for (int i = 0; i < 100; i++)
//     {
//         y_axis.at(i) = 0.1 * rand_gen.gaussian(W_SIGMA * W_SIGMA);
//     }
//     matplot::plot(y_axis, x_axis, "o");
//     matplot::grid(matplot::on);
//     matplot::show();
// }