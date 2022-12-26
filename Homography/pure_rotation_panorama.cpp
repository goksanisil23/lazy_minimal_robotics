#include "Homography.h"

#include "carla_img_reader.hpp"

int main()
{
    CarlaImageReader carlaImgReader("", "../resources/pure_rotation", "../resources/pure_rotation/camera_poses_gt.txt");

    homography::Homography homography(homography::CAMERA_TYPE::CARLA_1024_640_PINHOLE);

    cv::Mat      rgbImg;
    RoboticsPose camPose;
    int          img_ctr = 0;

    std::vector<cv::Mat> imgs;

    while (carlaImgReader.getNextImage(rgbImg))
    {
        imgs.emplace_back(rgbImg);
        img_ctr++;
    }

    cv::Mat result = homography.StitchRightToLeft(imgs.at(0), imgs.at(1));

    return 0;
}