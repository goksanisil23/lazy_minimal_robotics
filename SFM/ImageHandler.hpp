#pragma once

#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>

#include "cnpy.h"

class ImageHandler {
 public:
  virtual bool getNextImage(cv::Mat &image) = 0;
  virtual ~ImageHandler() {}
};

class NumpyImageHandler : public ImageHandler {
 public:
  NumpyImageHandler(const std::string &depthFilesDir,
                    const std::string &rgbFilesDir) {
    for (const auto &file :
         std::filesystem::directory_iterator(depthFilesDir)) {
      if (file.path().string().find("depth_") != std::string::npos) {
        _depthImageFiles.emplace_back(file.path().string());
      }
    }
    for (const auto &file : std::filesystem::directory_iterator(rgbFilesDir)) {
      if (file.path().string().find("rgb_") != std::string::npos) {
        _rgbImageFiles.emplace_back(file.path().string());
      }
    }
    // Sort the files alphabetically (chronologically)
    std::sort(_depthImageFiles.begin(), _depthImageFiles.end());
    std::sort(_rgbImageFiles.begin(), _rgbImageFiles.end());
    _depthImgsItr = _depthImageFiles.begin();
    _rgbImgsItr = _rgbImageFiles.begin();

    std::cout << "found " << _rgbImageFiles.size() << " rgb images\n";
    std::cout << "found " << _depthImageFiles.size() << " depth images\n";
  }

  template <typename T>
  cv::Mat getDepthImage(const std::string &depthImagePath) {
    // Load the data from file
    cnpy::NpyArray npyData = cnpy::npy_load(depthImagePath);
    // Get pointer to data
    T *npyDataPtr = npyData.data<T>();
    // Get the shape of data
    int imageHeight = npyData.shape[0];
    int imageWidth = npyData.shape[1];

    // // A) for viewing grayscale depth image
    // T minEl =
    //     *std::min_element(npyDataPtr, npyDataPtr + imageHeight * imageWidth);
    // T maxEl =
    //     *std::max_element(npyDataPtr, npyDataPtr + imageHeight * imageWidth);
    // cv::Mat depthImage(imageHeight, imageWidth, CV_8UC1, cv::Scalar(0));
    // for (size_t ii = 0; ii < imageHeight; ii++) {
    //   for (size_t jj = 0; jj < imageWidth; jj++) {
    //     // depthImage.at<uint8_t>(ii, jj) =
    //     //     static_cast<uint8_t>((npyDataPtr[jj + ii * imageWidth] -
    //     minEl) /
    //     //                          (maxEl * minEl) * 255.0);
    //     depthImage.at<uint8_t>(ii, jj) = static_cast<uint8_t>(
    //         (npyDataPtr[jj + ii * imageWidth] - minEl) / (maxEl)*255.0);
    //   }
    // }

    // B) For actually using the depth values
    double maxDepth = 0;
    cv::Mat depthImage(imageHeight, imageWidth, CV_64F, cv::Scalar(0.0));
    for (size_t ii = 0; ii < imageHeight; ii++) {
      for (size_t jj = 0; jj < imageWidth; jj++) {
        depthImage.at<double>(ii, jj) =
            static_cast<double>(npyDataPtr[jj + ii * imageWidth]);
        if (depthImage.at<double>(ii, jj) > maxDepth) {
          maxDepth = depthImage.at<double>(ii, jj);
        }
      }
    }

    _isCurrentDepthValid = (maxDepth == 0.0) ? false : true;

    return std::move(depthImage);
  }

  template <typename T>
  cv::Mat getRgbImage(std::string const &rgbImagePath) {
    // Load the data from file
    cnpy::NpyArray npyData = cnpy::npy_load(rgbImagePath);

    // Get pointer to data
    T *npyDataPtr = npyData.data<T>();
    T *itr = npyDataPtr;

    // Get the shape of data
    int imageHeight = npyData.shape[0];
    int imageWidth = npyData.shape[1];
    int channels = npyData.shape[2];

    cv::Mat cvImage(imageHeight, imageWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    size_t idx = 0;
    for (size_t ii = 0; ii < imageHeight; ii++) {
      for (size_t jj = 0; jj < imageWidth; jj++) {
        cv::Vec3b &cvColor = cvImage.at<cv::Vec3b>(ii, jj);
        cvColor[0] = *itr;
        itr++;
        cvColor[1] = *itr;
        itr++;
        cvColor[2] = *itr;
        itr++;
      }
    }
    return std::move(cvImage);
  }

  //   template <typename T>
  bool getNextImage(cv::Mat &image) {
    if (_rgbImgsItr != _rgbImageFiles.end()) {
      image = getRgbImage<uint8_t>(*_rgbImgsItr);
      _rgbImgsItr++;
      return true;
    } else {
      return false;
    }
  }

  template <typename T_rgb, typename T_depth>
  bool getNextImageWithDepth(cv::Mat &rgbImage, cv::Mat &depthImage) {
    if (_depthImgsItr != _depthImageFiles.end()) {
      rgbImage = getRgbImage<T_rgb>(*_rgbImgsItr);
      depthImage = getDepthImage<T_depth>(*_depthImgsItr);
      _depthImgsItr++;
      _rgbImgsItr++;
      return true;
    } else {
      return false;
    }
  }

  bool isCurrentDepthValid() { return _isCurrentDepthValid; }

  ~NumpyImageHandler() {}

 private:
  std::vector<std::string> _depthImageFiles, _rgbImageFiles;
  std::vector<std::string>::iterator _depthImgsItr, _rgbImgsItr;
  bool _isCurrentDepthValid;
};
