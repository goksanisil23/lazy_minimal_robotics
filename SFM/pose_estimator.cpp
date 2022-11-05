#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ImageHandler.hpp"
#include "PoseEstimator.hpp"

// const std::string IMAGES_DIR("/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/rgbd-pointcloud/imgs3");
const std::string IMAGES_DIR_RGB("../resources/data/imgs/rgb");
const std::string IMAGES_DIR_DEPTH("../resources/data/imgs/depth");

int main()
{
    // sfm::NumpyImageHandler imgHandler(IMAGES_DIR_RGB, IMAGES_DIR_DEPTH);
    sfm::CarlaImageHandler imgHandler(IMAGES_DIR_DEPTH, IMAGES_DIR_RGB);

    sfm::PoseEstimator pose_estimator;

    cv::Mat depthImg, rgbImg;
    int     img_ctr  = 0;
    int     img_step = 1;
    while (imgHandler.getNextImageWithDepth(rgbImg, depthImg))
    // imgHandler.getNextImageWithDepth<uint8_t, uint16_t>(rgbImg, depthImg))
    {
        if (imgHandler.isCurrentDepthValid())
        {
            // cv::imshow("depth", depthImg);
            // cv::waitKey(100);
            // cv::imshow("rgb", rgbImg);
            // cv::waitKey(100);
            // pose_estimator.projectImageTo3D(rgbImg, depthImg);
            pose_estimator.stepPnp(rgbImg, depthImg);
            img_ctr = (img_ctr + 1);
            std::cout << "img: " << img_ctr << std::endl;
        }
    }

    return 0;
}
