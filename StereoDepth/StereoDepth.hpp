#pragma once

#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "open3d/Open3D.h"

class StereoDepth
{
  public:
    // Type definitions
    typedef cv::Point3_<uint8_t> PixelDisparityU08;
    // typedef cv::Point<int16_t>   PixelDisparityS16Fixed;
    const float DEPTH_THRESHOLD = 999.0;

    // Constructor
    StereoDepth(cv::Mat intrinsics_K_left, cv::Mat intrinsics_K_right, Eigen::Matrix4f T_left_to_right)
        : K_left_(intrinsics_K_left), c_x_left_(K_left_.at<float>(0, 2)), c_y_left_(K_left_.at<float>(1, 2)),
          f_x_left_(K_left_.at<float>(0, 0)), f_y_left_(K_left_.at<float>(1, 1)), K_right_(intrinsics_K_right),
          c_x_right_(K_right_.at<float>(0, 2)), c_y_right_(K_right_.at<float>(1, 2)),
          f_x_right_(K_right_.at<float>(0, 0)), f_y_right_(K_right_.at<float>(1, 1)), T_left_to_right_(T_left_to_right)
    {
    }

    void projectLeftImgTo3D(const cv::Mat &rgbImageLeft, const cv::Mat &depthImageLeft)
    {
        static int                   imgIdx = 0;
        std::vector<Eigen::Vector3d> o3d_points;
        std::vector<Eigen::Vector3d> o3d_colors;

        for (int ii = 0; ii < rgbImageLeft.rows; ii++)
        {
            for (int jj = 0; jj < rgbImageLeft.cols; jj++)
            {
                float depth = depthImageLeft.at<float>(ii, jj);
                if ((depth > 0) && (depth < DEPTH_THRESHOLD)) // its possible that stereo depth returns 0
                {
                    float           x_world = (jj - c_x_left_) * depth / f_x_left_;
                    float           y_world = (ii - c_x_left_) * depth / f_y_left_;
                    float           z_world = depth;
                    Eigen::Vector3d pt_in_cam_frame(-x_world, -y_world, z_world);

                    // o3d_points.at(jj + ii * rgbImageLeft.cols) = pt_in_cam_frame;
                    o3d_points.push_back(pt_in_cam_frame);
                    auto rgbColor = rgbImageLeft.at<cv::Vec3b>(ii, jj);
                    o3d_colors.push_back(Eigen::Vector3d(rgbColor[2], rgbColor[1], rgbColor[0]) / 255.0);
                }
            }
        }
        // Write to file
        std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud;
        o3d_cloud          = std::make_shared<open3d::geometry::PointCloud>(o3d_points);
        o3d_cloud->points_ = o3d_points;
        o3d_cloud->colors_ = o3d_colors;
        open3d::io::WritePointCloud(std::to_string(imgIdx) + ".pcd", *o3d_cloud);
        imgIdx++;
    }

    cv::Mat computeDepthFromLeftDisparityMap(const cv::Mat &disparityMatS16Fixed)
    {
        // OpenCV's disparity map is fixed point 12.4 format. Convert to float
        constexpr float fixedS16ToFloat = 1.0f / static_cast<float>(1 << 4);
        // Z = b*f_x/disparity
        float baseline = T_left_to_right_(0, 3);

        cv::Mat depthMapF32(disparityMatS16Fixed.rows, disparityMatS16Fixed.cols, CV_32F, cv::Scalar::all(0));

        disparityMatS16Fixed.forEach<int16_t>(
            [&depthMapF32, baseline, this](int16_t &pixel, const int *position)
            {
                float disparity = static_cast<float>(pixel) * fixedS16ToFloat;
                if (disparity > 0)
                    depthMapF32.at<float>(position[0], position[1]) = baseline * f_x_left_ / disparity;
            });

        cv::Mat depth_U8;
        double  minVal, maxVal;
        cv::minMaxLoc(depthMapF32, &minVal, &maxVal);
        depthMapF32.convertTo(depth_U8, CV_8UC1, 255 / (maxVal - minVal));

        cv::imshow("depth", depth_U8);

        return depthMapF32;
    }

    cv::Mat computeLeftDisparityMapBM(const cv::Mat &leftImage, const cv::Mat &rightImage)
    {
        cv::Mat leftImageGray, rightImageGray;
        cv::cvtColor(leftImage, leftImageGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightImage, rightImageGray, cv::COLOR_BGR2GRAY);

        // (max_disparity-min_disparity), must be divisible by 16
        // min_disparity is the offset from the x-position of the left pixel at which search begins
        // for camera setups that are inclined towards each other, min_disparity can be negative
        int numDisparities = 6 * 16;
        // matching window size, must be odd.
        int blockSize = 11;

        auto leftMatcherBM = cv::StereoBM::create(numDisparities, blockSize);

        cv::Mat disparityLeft_S16Fixed;
        leftMatcherBM->compute(leftImageGray, rightImageGray, disparityLeft_S16Fixed);

        // Normalize for visualization
        cv::Mat disparityLeft_U8;
        double  minVal, maxVal;
        cv::minMaxLoc(disparityLeft_S16Fixed, &minVal, &maxVal);
        disparityLeft_S16Fixed.convertTo(disparityLeft_U8, CV_8UC1, 255 / (maxVal - minVal));

        cv::imshow("BM", disparityLeft_U8);
        // cv::waitKey(0);

        return disparityLeft_S16Fixed;
    }

    cv::Mat computeLeftDisparityMapSGBM(const cv::Mat &leftImage, const cv::Mat &rightImage)
    {
        cv::Mat leftImageGray, rightImageGray;
        cv::cvtColor(leftImage, leftImageGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightImage, rightImageGray, cv::COLOR_BGR2GRAY);

        // (max_disparity-min_disparity), must be divisible by 16
        // min_disparity is the offset from the x-position of the left pixel at which search begins
        // for camera setups that are inclined towards each other, min_disparity can be negative
        int numDisparities = 6 * 16;
        // matching window size, must be odd.
        int blockSize    = 11;
        int minDisparity = 0;
        int windowSize   = 6;

        auto leftMatcherSGBM = cv::StereoSGBM::create(minDisparity,
                                                      numDisparities,
                                                      blockSize,
                                                      8 * 3 * windowSize * windowSize,
                                                      32 * 3 * windowSize * windowSize,
                                                      1,
                                                      0,
                                                      10,
                                                      0,
                                                      0,
                                                      cv::StereoSGBM::MODE_SGBM_3WAY);

        cv::Mat disparityLeft_S16;
        leftMatcherSGBM->compute(leftImageGray, rightImageGray, disparityLeft_S16);

        // // Normalize for visualization
        cv::Mat disparityLeft_U8;
        double  minVal, maxVal;
        cv::minMaxLoc(disparityLeft_S16, &minVal, &maxVal);
        disparityLeft_S16.convertTo(disparityLeft_U8, CV_8UC1, 255 / (maxVal - minVal));

        cv::imshow("SGBM", disparityLeft_U8);
        // cv::waitKey(0);

        return disparityLeft_S16;
    }

  private:
    cv::Mat         K_left_; // camera intrinsics
    float           c_x_left_, c_y_left_, f_x_left_, f_y_left_;
    cv::Mat         K_right_; // camera intrinsics
    float           c_x_right_, c_y_right_, f_x_right_, f_y_right_;
    Eigen::Matrix4f T_left_to_right_; // extrinsics of right cam w.r.t left cam
};
