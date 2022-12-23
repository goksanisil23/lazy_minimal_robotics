#pragma once

#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>

namespace homography
{

// Feature matching parameters
constexpr int32_t MAX_ORB_FEAUTURES = 3000;
constexpr float   LOWE_MATCH_RATIO  = 0.5;
constexpr int     MIN_MATCHES       = 20;
constexpr float   RANSAC_ERR_THRES  = 2.0F;

class Homography
{
  public:
    Homography();

    void    StitchRightToLeft(const cv::Mat &leftImg, const cv::Mat &rightImg);
    cv::Mat FindHomography(const std::vector<cv::Point2d> &keypoints0, const std::vector<cv::Point2d> &keypoints1);

  private:
    // camera intrinsics
    double  cX_, cY_, fX_, fY_;
    cv::Mat K_;

    cv::Ptr<cv::Feature2D>         orb_;
    cv::Ptr<cv::Feature2D>         sift_;
    cv::Ptr<cv::DescriptorMatcher> flann_matcher_;
};
} // namespace homography