#pragma once

#include "Eigen/Dense"
#include "opencv2/core/eigen.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/viz.hpp>
#include <string>

#include "PlaneReference.hpp"

namespace homography
{
class Viz3D
{
  public:
    explicit Viz3D(const PlaneReference &planeRef, const cv::Matx33d &cameraIntrinsics);
    void Update(const cv::Mat &inputImg, const Eigen::Matrix4d &cameraPose);
    void Render();

  private:
    cv::viz::Viz3d cvViz3d_;
    cv::Matx33d    K_;

    std::mutex vizLock;
};

} // namespace homography