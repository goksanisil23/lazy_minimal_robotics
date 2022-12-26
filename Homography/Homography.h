#pragma once

#include "Eigen/Dense"
#include "opencv2/core/eigen.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>

#include "PlaneReference.hpp"

namespace homography
{

// Feature matching parameters
// constexpr int32_t MAX_FEAUTURES = 3000;
constexpr int32_t MAX_FEAUTURES = 1000;
// constexpr float   LOWE_MATCH_RATIO  = 0.5;
constexpr float LOWE_MATCH_RATIO = 0.6;
constexpr int   MIN_MATCHES      = 15;
// constexpr float RANSAC_ERR_THRES = 4.0F;
constexpr float RANSAC_ERR_THRES = 2.0F;

enum class CAMERA_TYPE
{
    CARLA_1024_640_PINHOLE = 1,
    OAK_D_LEFT             = 2,
    OAK_D_RIGHT            = 3,
    OAD_D_CENTER           = 4
};

enum class HOMOGRAPHY_MODE
{
    IMAGE_TO_IMAGE = 1,
    OBJ_TO_IMAGE   = 2
};

class Homography
{
  public:
    Homography(const CAMERA_TYPE &cameraType);

    void    FindMatches(const cv::Mat &dstImg, const cv::Mat &srcImg);
    cv::Mat ComputeHomography(const HOMOGRAPHY_MODE &homMode);
    cv::Mat UndistortImage(const cv::Mat &inputImg);

    cv::Mat StitchRightToLeft(const cv::Mat &leftImg, const cv::Mat &rightImg);

    void
    LocateTemplatePlane(const cv::Mat &planeReferenceImg, const cv::Mat &inputImage, const cv::Mat &homographyMatrix);
    void            SetPlaneReferenceImage(const cv::Mat &planeReferenceImg,
                                           const double  &planeReferenceObjWidth  = 0,
                                           const double  &planeReferenceObjHeight = 0);
    Eigen::Matrix4d ComputeCameraPose(const cv::Mat &homographyMatrix3d);

    PlaneReference GetPlaneRefCopy() const;
    cv::Matx33d    GetCameraInstrinsics() const;

  private:
    // camera intrinsics
    double          cX_, cY_, fX_, fY_;
    cv::Mat         K_;
    Eigen::Matrix3d KEigen_;
    cv::Mat         distCoeffs_;

    cv::Ptr<cv::Feature2D>         orb_;
    cv::Ptr<cv::Feature2D>         sift_;
    cv::Ptr<cv::DescriptorMatcher> flann_matcher_;

    std::vector<cv::Point2d> goodKeypointsSrc_;
    std::vector<cv::Point2d> goodKeypointsDst_;
    std::vector<size_t>      goodKeypointsSrcIdxs_;

    std::unique_ptr<PlaneReference> planeReference_;
};
} // namespace homography