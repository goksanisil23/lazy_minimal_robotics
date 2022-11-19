#include <fstream>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "FeatureMatcher.hpp"
#include "ImageHandler.hpp"

const std::string IMAGES_RGB_DIR("../resources/data/imgs/rgb");
const std::string IMAGES_DEPTH_DIR("../resources/data/imgs/depth");
const std::string CAMERA_POSES_PATH("../resources/data/camera_poses_odom.txt");

sfm::FeatureMatcher::RoboticsPose getCameraPoseFromString(const std::string cameraPoseString)
{
    std::stringstream stream(cameraPoseString);
    float             x, y, z, qx, qy, qz, qw;
    int               imgIdx;
    stream >> imgIdx >> x >> y >> z >> qx >> qy >> qz >> qw;

    return sfm::FeatureMatcher::RoboticsPose(x, y, z, qx, qy, qz, qw);
}

int main()
{
    // sfm::NumpyImageHandler imgHandler(IMAGES_DIR_RGB, IMAGES_DIR_DEPTH);
    sfm::CarlaImageHandler imgHandler(IMAGES_DEPTH_DIR, IMAGES_RGB_DIR);
    sfm::FeatureMatcher    feature_matcher;

    std::fstream odomCameraPosesTxtStream;
    odomCameraPosesTxtStream.open(CAMERA_POSES_PATH);
    std::string cameraOdomLine;
    std::getline(odomCameraPosesTxtStream, cameraOdomLine); // skip the 1st header line

    cv::Mat depthImg, rgbImg;
    int     img_ctr  = 0;
    int     img_step = 1;
    while (imgHandler.getNextImageWithDepth(rgbImg, depthImg))
    {
        // Get the camera pose
        std::getline(odomCameraPosesTxtStream, cameraOdomLine);
        sfm::FeatureMatcher::RoboticsPose cameraPose(getCameraPoseFromString(cameraOdomLine));

        if (imgHandler.isCurrentDepthValid())
        {
            feature_matcher.AddRgbDepthPair(rgbImg, depthImg);
            feature_matcher.AddCameraPose(cameraPose);
            img_ctr = (img_ctr + 1);
            std::cout << "Read img: " << img_ctr << std::endl;

            // if (img_ctr == 50)
            //     break;
        }
    }

    feature_matcher.FindGlobalMatches();

    feature_matcher.CreateBaDataset();

    return 0;
}
