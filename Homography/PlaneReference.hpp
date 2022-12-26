#pragma once

#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>

namespace homography
{

// Origin of the 3D world defined by the plane reference is chosen to be top left corner of the reference image
class PlaneReference
{
  public:
    PlaneReference(const cv::Mat &planeReferenceImg,
                   const double  &planeReferenceObjWidth  = 0,
                   const double  &planeReferenceObjHeight = 0)
        : referenceImg_{planeReferenceImg}, planeReferenceObjWidth_{planeReferenceObjWidth},
          planeReferenceObjHeight_{planeReferenceObjHeight}
    {
        if (planeReferenceObjWidth)
        {
            metersPerPixelU_ = planeReferenceObjWidth / static_cast<double>(planeReferenceImg.size().width);
            metersPerPixelV_ = planeReferenceObjHeight / static_cast<double>(planeReferenceImg.size().height);
        }

        // Detect the keypoints and descriptors of the reference image
        cv::cvtColor(planeReferenceImg, referenceImgGray_, cv::COLOR_RGB2GRAY);
    }

    inline void GenerateFeatureDescriptors(const cv::Ptr<cv::Feature2D> featureGenerator)
    {
        featureGenerator->detectAndCompute(referenceImgGray_, cv::Mat(), keypoints_, descriptors_);
        descriptors_.convertTo(descriptors_, CV_32F); // descriptors are integers but for knn we neet float

        std::cout << "Generated " << keypoints_.size() << " descriptors for the reference plane object" << std::endl;

        // Compute the world points
        for (const auto &kp : keypoints_)
        {
            worldPoints_.emplace_back(PixelToWorldCoordinates(kp.pt));
        }
    }

    inline cv::Point3d PixelToWorldCoordinates(const cv::Point2f &pixelCoords) const
    {
        // Set Z=0
        return pixelCoords.x * metersPerPixelU_ * cv::Point3d{1.0f, 0.0f, 0.0f} +
               pixelCoords.y * metersPerPixelV_ * cv::Point3d{0.0f, 1.0f, 0.0f};
    }

    cv::Mat referenceImg_;
    cv::Mat referenceImgGray_;
    double  metersPerPixelU_;
    double  metersPerPixelV_;
    double  planeReferenceObjWidth_;
    double  planeReferenceObjHeight_;

    std::vector<cv::KeyPoint> keypoints_;
    std::vector<cv::Point3d>  worldPoints_;
    cv::Mat                   descriptors_; // (num_keypoints x descriptor_size) = (3000 x 32)
};
} // namespace homography