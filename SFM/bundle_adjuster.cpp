#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "open3d/Open3D.h"
#include <opencv4/opencv2/opencv.hpp>

#include <g2o/core/optimization_algorithm_levenberg.h>

#include "BaDataParser.hpp"
#include "BundleAdjuster.hpp"

const std::string BA_DATASET_PATH{"/home/goksan/Work/lazy_minimal_robotics/SFM/resources/data/data_for_ba.txt"};

int main()
{
    // Parse the dataset
    std::vector<ba::dataset_parser::Observation>  measurements;
    std::vector<ba::dataset_parser::RoboticsPose> cameraPoses;
    std::vector<Eigen::Vector3d>                  landmarkPositions;
    parseBaDataset(BA_DATASET_PATH, measurements, cameraPoses, landmarkPositions);
    std::cout << measurements.size() << " " << cameraPoses.size() << " " << landmarkPositions.size() << std::endl;

    // Construct g2o optimization problem

    // pose dimension 6, landmark dimension 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    // Linear solver with the residual and parameter size defined above
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    //   Specify the gradient descent method
    auto g2o_solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer g2o_optimizer;
    g2o_optimizer.setAlgorithm(g2o_solver);
    g2o_optimizer.setVerbose(true);

    // Build the problem
    std::vector<ba::Pose6dVertex *>  cameraPoseVertices;
    std::vector<ba::Point3dVertex *> landmarkPositionVertices;

    // Add the camera pose vertices to the graph
    for (int i = 0; i < cameraPoses.size(); i++)
    {
        ba::dataset_parser::RoboticsPose cameraPose = cameraPoses.at(i);

        ba::Pose6dVertex *pose6dVertex = new ba::Pose6dVertex();
        pose6dVertex->setId(i);
        pose6dVertex->setEstimate(
            ba::Pose6d(Eigen::Vector3d(cameraPose.x, cameraPose.y, cameraPose.z),
                       Eigen::Quaterniond(cameraPose.qw, cameraPose.qx, cameraPose.qy, cameraPose.qz)));
        g2o_optimizer.addVertex(pose6dVertex);
        cameraPoseVertices.push_back(pose6dVertex);
    }

    // Add landmark vertices to the graph
    for (int i = 0; i < landmarkPositions.size(); i++)
    {
        Eigen::Vector3d landmarkPosition = landmarkPositions.at(i);

        ba::Point3dVertex *point3dVertex = new ba::Point3dVertex();
        point3dVertex->setId(cameraPoseVertices.size() + i);
        point3dVertex->setEstimate(landmarkPosition);
        point3dVertex->setMarginalized(true);
        g2o_optimizer.addVertex(point3dVertex);
        landmarkPositionVertices.push_back(point3dVertex);
    }

    // Add edges(observations/measurements) to the graph
    for (int i = 0; i < measurements.size(); i++)
    {
        ba::dataset_parser::Observation measurement = measurements.at(i);

        ba::PerspectiveProjectionEdge *edge = new ba::PerspectiveProjectionEdge;
        edge->setVertex(0, cameraPoseVertices.at(measurement.cameraIdx));         // camera that sees this landmark
        edge->setVertex(1, landmarkPositionVertices.at(measurement.landmarkIdx)); // the landmark that camera sees
        edge->setMeasurement(Eigen::Vector2d(
            measurement.keypointU, measurement.keypointV)); // pixel coordinates of this landmark on this camera image
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        g2o_optimizer.addEdge(edge);
    }

    // Run the optimization
    g2o_optimizer.initializeOptimization();
    g2o_optimizer.optimize(1000);

    // // Retrieve the results
    // std::vector<double> finalPoses;
    // double              rmse_before{0}, rmse_after{0};
    // for (int i = 0; i < poseVertices.size(); i++)
    // {
    //     std::cout << "robot: " << i << " pos: " << poseVertices.at(i)->estimate() << std::endl;
    //     finalPoses.push_back(poseVertices.at(i)->estimate());
    //     rmse_before += std::pow(y_gt.at(i) - y_odom.at(i), 2);
    //     rmse_after += std::pow(y_gt.at(i) - finalPoses.at(i), 2);
    // }

    return 0;
}
