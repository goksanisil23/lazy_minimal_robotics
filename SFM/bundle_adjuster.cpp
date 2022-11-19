#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "open3d/Open3D.h"
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <matplot/matplot.h>
#include <opencv4/opencv2/opencv.hpp>

#include "BaDataParser.hpp"
#include "BundleAdjuster.hpp"

// File paths
const std::string BA_DATASET_PATH{"/home/goksan/Work/lazy_minimal_robotics/SFM/resources/data/data_for_ba.txt"};
const std::string OPT_CAM_POSES_OUT_PATH{
    "/home/goksan/Work/lazy_minimal_robotics/SFM/resources/data/camera_poses_opt.txt"};
const std::string OPT_LANDMARKS_OUT_PATH{
    "/home/goksan/Work/lazy_minimal_robotics/SFM/resources/data/landmark_positions_opt.txt"};

// Bundle adjustment parameters
const int N_ITER{200};

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
        // if (i == 0) // 1st camera pose is fixed
        //     pose6dVertex->setFixed(true);
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
        // edge->setRobustKernel(new g2o::RobustKernelHuber());
        // edge->setRobustKernel(new g2o::RobustKernelWelsch()); // diverges after some iterations, but at lowest chi^2, gives good result
        // edge->setRobustKernel(new g2o::RobustKernelCauchy()); // MUCH BETTER THAN HUBER and TUKEY
        edge->setRobustKernel(new g2o::RobustKernelGemanMcClure()); // WOrks best for 20+ images
        // edge->setRobustKernel(nullptr); // Causes all poses to be same. Weird.
        g2o_optimizer.addEdge(edge);
    }

    // Run the optimization
    g2o_optimizer.initializeOptimization();
    g2o_optimizer.optimize(N_ITER);

    // Retrieve the results

    std::vector<double> cam_x_opt, cam_z_opt;
    for (int i = 0; i < cameraPoseVertices.size(); i++)
    {
        ba::Pose6d cameraPoseRefined = cameraPoseVertices.at(i)->estimate();
        cam_x_opt.push_back(cameraPoseRefined.translation(0));
        cam_z_opt.push_back(cameraPoseRefined.translation(2));
    }
    // Unoptimized camera poses
    std::vector<double> cam_x, cam_z;
    for (const ba::dataset_parser::RoboticsPose &camPose : cameraPoses)
    {
        cam_x.push_back(camPose.x);
        cam_z.push_back(camPose.z);
    }

    // Save the optimized camera poses, and landmarks
    std::ofstream optCamPoseOut;
    optCamPoseOut.open(OPT_CAM_POSES_OUT_PATH);
    optCamPoseOut << "image_index t_x t_y t_z q_x q_y q_z q_w" << std::endl;
    for (int i = 0; i < cameraPoseVertices.size(); i++)
    {
        ba::Pose6d cameraPoseRefined = cameraPoseVertices.at(i)->estimate();
        optCamPoseOut << i << " " << cameraPoseRefined.translation(0) << " " << cameraPoseRefined.translation(1) << " "
                      << cameraPoseRefined.translation(2) << " " << cameraPoseRefined.rotation.unit_quaternion().x()
                      << " " << cameraPoseRefined.rotation.unit_quaternion().y() << " "
                      << cameraPoseRefined.rotation.unit_quaternion().z() << " "
                      << cameraPoseRefined.rotation.unit_quaternion().w() << std::endl;
    }
    optCamPoseOut.close();

    std::ofstream optLandmarkPosOut;
    optLandmarkPosOut.open(OPT_LANDMARKS_OUT_PATH);
    for (int i = 0; i < landmarkPositionVertices.size(); i++)
    {
        Eigen::Vector3d landmarkPos = landmarkPositionVertices.at(i)->estimate();
        optLandmarkPosOut << landmarkPos(0) << " " << landmarkPos(1) << " " << landmarkPos(2) << std::endl;
    }
    optLandmarkPosOut.close();

    // Plot
    matplot::cla();
    matplot::hold(matplot::on);
    matplot::plot(cam_z, cam_x, "o");
    matplot::plot(cam_z_opt, cam_x_opt, "s");
    matplot::legend();
    matplot::grid(matplot::on);
    matplot::show();

    return 0;
}
