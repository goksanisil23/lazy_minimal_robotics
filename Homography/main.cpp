#include "Homography.h"

#include "carla_img_reader.hpp"

int main()
{
    CarlaImageReader carlaImgReader("", "../resources/pure_rotation", "../resources/pure_rotation/camera_poses_gt.txt");

    homography::Homography homography;

    cv::Mat      rgbImg;
    RoboticsPose camPose;
    int          img_ctr = 0;

    std::vector<cv::Mat> imgs;

    // while (carlaImgReader.getNextImageWithCamPose(rgbImg, camPose))
    while (carlaImgReader.getNextImage(rgbImg))
    {
        {
            imgs.emplace_back(rgbImg);
        }
        img_ctr++;
    }

    homography.StitchRightToLeft(imgs.at(0), imgs.at(1));
    // homography.ComputeHomography(imgs.at(0), imgs.at(2));
    // homography.ComputeHomography(imgs.at(1), imgs.at(0));

    return 0;
}