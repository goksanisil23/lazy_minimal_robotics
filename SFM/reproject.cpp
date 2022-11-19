#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ImageHandler.hpp"

#include "open3d/Open3D.h"
#include <opencv4/opencv2/opencv.hpp>

const std::string RGB_PCD_PATH_PREFIX = "../resources/data/pcds/after_g2o/rgb/rgb_cloud_";
const std::string CAM_POSE_PATH       = "../resources/data/camera_poses_opt.txt";
const std::string IMAGES_DIR_RGB("../resources/data/imgs/rgb");
const std::string IMAGES_DIR_DEPTH("../resources/data/imgs/depth");

constexpr double DEPTH_THRESHOLD = 2000.0;

// Camera parameters
constexpr double _cX = 512.0f;
constexpr double _cY = 320.0f;
constexpr double _fX = 512.0f;
constexpr double _fY = 512.0f;

void inline transformPoint(const Eigen::Matrix3d &rot, const Eigen::Vector3d &trans, Eigen::Vector3d &pt)
{
    auto t1 = pt(0);
    auto t2 = pt(1);
    auto t3 = pt(2);

    pt(0) = rot(0, 0) * t1 + rot(0, 1) * t2 + rot(0, 2) * t3 + trans(0);
    pt(1) = rot(1, 0) * t1 + rot(1, 1) * t2 + rot(1, 2) * t3 + trans(1);
    pt(2) = rot(2, 0) * t1 + rot(2, 1) * t2 + rot(2, 2) * t3 + trans(2);
}

void projectImageTo3d(const int             &imgIdx,
                      const cv::Mat         &rgbImage,
                      const cv::Mat         &depthImage,
                      const Eigen::Matrix3d &rot,
                      const Eigen::Vector3d &trans)
{
    std::vector<Eigen::Vector3d> o3d_points;
    std::vector<Eigen::Vector3d> o3d_colors;

    for (int ii = 0; ii < rgbImage.rows; ii++)
    {
        for (int jj = 0; jj < rgbImage.cols; jj++)
        {
            double depth = depthImage.at<double>(ii, jj);
            if ((depth > 0) && (depth < DEPTH_THRESHOLD)) // its possible that stereo depth returns 0
            {
                double          x_world = (jj - _cX) * depth / _fX;
                double          y_world = (ii - _cY) * depth / _fY;
                double          z_world = depth;
                Eigen::Vector3d pt_in_cam_frame(-x_world, -y_world, z_world);
                Eigen::Vector3d pt_in_world_frame = pt_in_cam_frame;
                transformPoint(rot, trans, pt_in_world_frame);

                // o3d_points.at(jj + ii * rgbImage.cols) = pt_in_cam_frame;
                o3d_points.push_back(pt_in_world_frame);
                auto rgbColor = rgbImage.at<cv::Vec3b>(ii, jj);
                o3d_colors.push_back(Eigen::Vector3d(rgbColor[2], rgbColor[1], rgbColor[0]) / 255.0);
            }
        }
    }
    // Write to file
    std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud;
    o3d_cloud          = std::make_shared<open3d::geometry::PointCloud>(o3d_points);
    o3d_cloud->points_ = o3d_points;
    o3d_cloud->colors_ = o3d_colors;
    open3d::io::WritePointCloud(RGB_PCD_PATH_PREFIX + std::to_string(imgIdx) + ".pcd", *o3d_cloud);
}

int main()
{
    // Get images
    sfm::CarlaImageHandler imgHandler(IMAGES_DIR_DEPTH, IMAGES_DIR_RGB);
    // Get camera poses
    std::ifstream odom_file(CAM_POSE_PATH);

    cv::Mat     depthImg, rgbImg;
    int         img_ctr = 0;
    std::string line;
    std::getline(odom_file, line); // 1st line is header
    while (std::getline(odom_file, line))
    {
        imgHandler.getNextImageWithDepth(rgbImg, depthImg);

        std::istringstream istream(line);
        int                imgIdx;
        double             tx, ty, tz, qx, qy, qz, qw;
        istream >> imgIdx;
        istream >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Eigen::Vector3d    camPosition(tx, ty, tz);
        Eigen::Quaterniond camRotation(qw, qx, qy, qz);

        if (imgHandler.isCurrentDepthValid())
        {
            projectImageTo3d(img_ctr, rgbImg, depthImg, camRotation.toRotationMatrix(), camPosition);
            img_ctr++;
            std::cout << "img: " << img_ctr << std::endl;
        }
    }
}