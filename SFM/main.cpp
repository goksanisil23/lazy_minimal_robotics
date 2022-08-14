#include <matplot/matplot.h>

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ImageHandler.hpp"
#include "Tracker.hpp"

int main() {
  NumpyImageHandler imgHandler(
      "/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/"
      "rgbd-pointcloud/imgs",
      "/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/"
      "rgbd-pointcloud/imgs");

  Tracker tracker;

  cv::Mat depthImg, rgbImg;
  while (
      imgHandler.getNextImageWithDepth<uint8_t, uint16_t>(rgbImg, depthImg)) {
    // cv::imshow("depth", depthImg);
    // cv::waitKey(100);
    // cv::imshow("rgb", rgbImg);
    // cv::waitKey(100);

    std::cout << imgHandler.isCurrentDepthValid() << " ";

    // tracker.projectImageTo3D(rgbImg, depthImg);
  }

  return 0;
}
