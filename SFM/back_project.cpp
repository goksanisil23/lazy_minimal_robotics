#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "open3d/Open3D.h"
#include <opencv4/opencv2/opencv.hpp>

void backProject()
{
    // Read pointcloud
    open3d::geometry::PointCloud point_cloud;
    open3d::io::ReadPointCloud(
        "/home/goksan/Work/lazy_minimal_robotics/SFM/resources/data/pcds/after_viso/rgb/rgb_cloud_20.pcd", point_cloud);

    // Get camera pose for this cloud
    std::ifstream odom_file("/home/goksan/Work/lazy_minimal_robotics/SFM/resources/data/camera_poses_odom.txt");
    int           line_ctr = 1;
    std::string   line;
    while (line_ctr != 22)
    {
        std::getline(odom_file, line);
        line_ctr++;
    }
    std::getline(odom_file, line);
    std::istringstream istream(line);
    int                imgIdx;
    double             tx, ty, tz, qx, qy, qz, qw;
    istream >> imgIdx;
    istream >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

    // Back project, using camera intrinsics
    double _cX = 512.0f;
    double _cY = 320.0f;
    double _fX = 512.0f;
    double _fY = 512.0f;

    // A)
    // Create transformation matrix for (T_camera_to_world) ^-1
    Eigen::Matrix4d T_w_c(Eigen::Matrix4d::Identity());
    T_w_c(0, 3)             = tx;
    T_w_c(1, 3)             = ty;
    T_w_c(2, 3)             = tz;
    T_w_c.block<3, 3>(0, 0) = Eigen::Quaterniond(qw, qx, qy, qz).matrix();
    Eigen::Matrix4d T_c_w   = T_w_c.inverse();

    // B)
    // Perspective projection matrix
    // Extrinsics
    Eigen::Matrix<double, 3, 4> P_ext;
    P_ext.block<3, 3>(0, 0) = Eigen::Quaterniond(qw, qx, qy, qz).matrix();
    P_ext(0, 3)             = tx;
    P_ext(1, 3)             = ty;
    P_ext(2, 3)             = tz;
    // Intrinsics
    Eigen::Matrix3d P_intr(Eigen::Matrix3d::Identity());
    P_intr(0, 0)                            = _fX;
    P_intr(1, 1)                            = _fY;
    P_intr(0, 2)                            = _cX;
    P_intr(1, 2)                            = _cY;
    Eigen::Matrix<double, 3, 4> P_pers_proj = P_intr * P_ext;

    cv::Mat image(640, 1024, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < point_cloud.points_.size(); i++)
    {
        // A) 3d Point in world coordinate frame
        Eigen::Matrix4d P_w(Eigen::Matrix4d::Identity());
        P_w(0, 3) = point_cloud.points_.at(i)(0);
        P_w(1, 3) = point_cloud.points_.at(i)(1);
        P_w(2, 3) = point_cloud.points_.at(i)(2);
        // 3d Point in camera coordinate frame
        Eigen::Matrix4d P_c(Eigen::Matrix4d::Identity());
        P_c = T_c_w * P_w;
        // project to image using intrinsics
        int u = static_cast<int>(std::round(_fX * -P_c(0, 3) / P_c(2, 3) + _cX));
        int v = static_cast<int>(std::round(_fY * -P_c(1, 3) / P_c(2, 3) + _cY));

        // B)
        // Eigen::Vector4d P_w;
        // P_w(0) = point_cloud.points_.at(i)(0);
        // P_w(1) = point_cloud.points_.at(i)(1);
        // P_w(2) = point_cloud.points_.at(i)(2);
        // P_w(3) = 1;
        // Eigen::Vector3d P_image = P_pers_proj * P_w;
        // int u = static_cast<int>(std::round(P_image(0) / P_image(2)));
        // int v = static_cast<int>(std::round(P_image(1) / P_image(2)));
        // std::cout << P_image(0) << " " << P_image(1) << " " << P_image(2) << std::endl;

        if (u > 0 && v > 0)
        {
            cv::Vec3b &cv_color = image.at<cv::Vec3b>(v, u);
            cv_color[0]         = static_cast<uint8_t>(std::round(point_cloud.colors_.at(i)(2) * 255.0));
            cv_color[1]         = static_cast<uint8_t>(std::round(point_cloud.colors_.at(i)(1) * 255.0));
            cv_color[2]         = static_cast<uint8_t>(std::round(point_cloud.colors_.at(i)(0) * 255.0));
        }
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}