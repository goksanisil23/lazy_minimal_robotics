#pragma once

#include <opencv4/opencv2/opencv.hpp>

#include "open3d/Open3D.h"

class Tracker {
 public:
  Tracker() {
    _cX = 1917.1802978515625;
    _cY = 1075.5216064453125;
    _fX = 3094.48486328125;
    _fY = 3094.48486328125;

    const double imageHeightOriginal = 2160;
    const double imageWidthOriginal = 3840;

    // const double imageHeight = 120;
    // const double imageWidth = 160;
    const double imageHeight = 360;
    const double imageWidth = 640;

    const double aspRatioX = imageWidthOriginal / imageHeight;
    const double aspRatioY = imageWidthOriginal / imageWidth;

    _cX = _cX / aspRatioX;
    _cY = _cY / aspRatioY;
    _fX = _fX / aspRatioX;
    _fY = _fY / aspRatioY;

    // Initialize pointcloud visualizer
    // setup visualization
    vis.CreateVisualizerWindow("projected camera", 960, 540, 480, 270);
    vis.GetRenderOption().background_color_ = {0.05, 0.05, 0.05};
    vis.GetRenderOption().point_size_ = 3;
    vis.GetRenderOption().show_coordinate_frame_ = true;
  }

  void projectImageTo3D(const cv::Mat &rgbImage, const cv::Mat &depthImage) {
    static int idx = 0;
    std::vector<Eigen::Vector3d> o3d_points(rgbImage.rows * rgbImage.cols);
    std::vector<Eigen::Vector3d> o3d_colors(rgbImage.rows * rgbImage.cols);

    for (int ii = 0; ii < rgbImage.rows; ii++) {
      for (int jj = 0; jj < rgbImage.cols; jj++) {
        double depth = depthImage.at<double>(ii, jj);
        double x_world = (jj - _cX) * depth / _fX;
        double y_world = (ii - _cY) * depth / _fY;
        double z_world = depth;
        o3d_points.at(jj + ii * rgbImage.cols) =
            Eigen::Vector3d(x_world, -y_world, z_world);
        auto rgbColor = rgbImage.at<cv::Vec3b>(ii, jj);
        o3d_colors.at(jj + ii * rgbImage.cols) =
            Eigen::Vector3d(rgbColor[2], rgbColor[1], rgbColor[0]) / 255.0;
      }
    }
    // Update visualizers
    if (idx == 0) {
      o3d_cloud = std::make_shared<open3d::geometry::PointCloud>(o3d_points);
      vis.AddGeometry(o3d_cloud);
      o3d_cloud->colors_.resize(o3d_cloud->points_.size(),
                                Eigen::Vector3d(0, 0, 0));
    } else {
      o3d_cloud->points_ = o3d_points;
      o3d_cloud->colors_ = o3d_colors;
      vis.UpdateGeometry();
      vis.PollEvents();
      vis.UpdateRender();
    }
    open3d::io::WritePointCloud("pointcloud_" + std::to_string(idx) + ".ply",
                                *o3d_cloud);
    idx++;
  }

 private:
  double _cX, _cY, _fX, _fY;
  open3d::visualization::Visualizer vis;
  std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud;
};