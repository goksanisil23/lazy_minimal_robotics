#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "StereoDepth.hpp"

int main()
{
    cv::Mat leftImg  = cv::imread("../resources/middlebury_motorcycle_left.png");
    cv::Mat rightImg = cv::imread("../resources/middlebury_motorcycle_right.png");

    int scaleRatio       = 4;
    int heightScaledDown = leftImg.rows / scaleRatio;
    int widthScaledDown  = leftImg.cols / scaleRatio;
    cv::resize(leftImg, leftImg, cv::Size(widthScaledDown, heightScaledDown));
    cv::resize(rightImg, rightImg, cv::Size(widthScaledDown, heightScaledDown));

    // Intrinsics of the dataset is obtained from https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Motorcycle-perfect/calib.txt
    cv::Mat K_left  = (cv::Mat_<float>(3, 3) << 3979.911, 0, 1244.772, 0, 3979.911, 1019.507, 0, 0, 1);
    cv::Mat K_right = (cv::Mat_<float>(3, 3) << 3979.911, 0, 1369.115, 0, 3979.911, 1019.507, 0, 0, 1);
    K_left.at<float>(0, 0) /= scaleRatio;
    K_left.at<float>(0, 2) /= scaleRatio;
    K_left.at<float>(1, 1) /= scaleRatio;
    K_left.at<float>(1, 2) /= scaleRatio;
    K_right.at<float>(0, 0) /= scaleRatio;
    K_right.at<float>(0, 2) /= scaleRatio;
    K_right.at<float>(1, 1) /= scaleRatio;
    K_right.at<float>(1, 2) /= scaleRatio;

    // Extrinsics are specified in OpenCV convention: X: right, Y: down, Z: outwards from camera
    Eigen::Matrix4f extrinsics_left_to_right_cam(Eigen::Matrix4f::Identity());
    extrinsics_left_to_right_cam(0, 3) = 193.001 / 1000.0; // x: right
    extrinsics_left_to_right_cam(1, 3) = 0;                // y: down
    extrinsics_left_to_right_cam(2, 3) = 0;                // z: up

    StereoDepth stereo_depth(K_left, K_right, extrinsics_left_to_right_cam);

    auto disparityLeftBM   = stereo_depth.computeLeftDisparityMapBM(leftImg, rightImg);
    auto disparityLeftSGBM = stereo_depth.computeLeftDisparityMapSGBM(leftImg, rightImg, true);
    auto depthMap          = stereo_depth.computeDepthFromLeftDisparityMap(disparityLeftSGBM);
    // Use Stereo-depth for 3D projection
    stereo_depth.projectLeftImgTo3D(leftImg, depthMap);

    cv::waitKey(0);

    return 0;
}