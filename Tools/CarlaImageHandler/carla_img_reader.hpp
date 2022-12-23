#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>

#include "utils.h"

class CarlaImageReader
{
  public:
    CarlaImageReader(const std::string &depthFilesDir,
                     const std::string &rgbFilesDir,
                     const std::string &cameraPoseFilePath)
    {
        ParseCameraPoses(cameraPoseFilePath);
        std::cout << "parsed " << camPoses_.size() << " camera poses" << std::endl;

        if (!depthFilesDir.empty())
        {
            for (const auto &file : std::filesystem::directory_iterator(depthFilesDir))
            {
                if (file.path().string().find("depth_") != std::string::npos)
                {
                    _depthImageFiles.emplace_back(file.path().string());
                }
            }
        }

        if (!rgbFilesDir.empty())
        {
            for (const auto &file : std::filesystem::directory_iterator(rgbFilesDir))
            {
                if (file.path().string().find("rgb_") != std::string::npos)
                {
                    _rgbImageFiles.emplace_back(file.path().string());
                }
            }
        }

        // Sort the files alphabetically (chronologically)
        std::sort(_depthImageFiles.begin(), _depthImageFiles.end());
        std::sort(_rgbImageFiles.begin(), _rgbImageFiles.end());
        _depthImgsItr = _depthImageFiles.begin();
        _rgbImgsItr   = _rgbImageFiles.begin();

        std::cout << "found " << _rgbImageFiles.size() << " rgb images\n";
        std::cout << "found " << _depthImageFiles.size() << " depth images\n";
    }

    cv::Mat getDepthImage(const std::string &depthImagePath)
    {
        cv::Mat depthImage = cv::imread(depthImagePath);
        cv::Mat depthImgF64(depthImage.rows, depthImage.cols, CV_64F, cv::Scalar(0.0));

        for (int ii = 0; ii < depthImage.rows; ii++)
        {
            for (int jj = 0; jj < depthImage.cols; jj++)
            {
                cv::Vec3b &cv_color       = depthImage.at<cv::Vec3b>(ii, jj);
                double    &depth_at_pixel = depthImgF64.at<double>(ii, jj);
                depth_at_pixel = static_cast<double>(cv_color[2] + cv_color[1] * 256 + cv_color[0] * 256 * 256) /
                                 static_cast<double>(256 * 256 * 256 - 1) * 1000.0;
            }
        }

        _isCurrentDepthValid = true;

        return std::move(depthImgF64);
    }

    bool getNextImageWithDepth(cv::Mat &rgbImage, cv::Mat &depthImage)
    {
        if (_depthImgsItr != _depthImageFiles.end())
        {
            rgbImage   = getRgbImage(*_rgbImgsItr);
            depthImage = getDepthImage(*_depthImgsItr);
            _depthImgsItr++;
            _rgbImgsItr++;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool getNextImageWithCamPose(cv::Mat &rgbImage, RoboticsPose &camPose)
    {
        static auto camPoseItr = camPoses_.begin();

        if (_rgbImgsItr != _rgbImageFiles.end())
        {
            rgbImage = getRgbImage(*_rgbImgsItr);
            _rgbImgsItr++;
            camPose = *camPoseItr;
            camPoseItr++;
            return true;
        }
        else
        {
            return false;
        }
    }

    void ParseCameraPoses(const std::string &cameraPoseFilePath)
    {
        // Get camera pose for this cloud
        std::ifstream odom_file(cameraPoseFilePath);
        std::string   line;
        while (std::getline(odom_file, line))
        {
            std::istringstream istream(line);
            int                imgIdx;
            RoboticsPose       camPose;
            double             tx, ty, tz, qx, qy, qz, qw;
            istream >> imgIdx;
            istream >> camPose.x >> camPose.y >> camPose.z >> camPose.qx >> camPose.qy >> camPose.qz >> camPose.qw;
            camPoses_.emplace_back(camPose);
        }
    }

    cv::Mat getRgbImage(std::string const &rgbImagePath)
    {
        cv::Mat cvImage = cv::imread(rgbImagePath);
        return std::move(cvImage);
    }

    bool getNextImage(cv::Mat &image)
    {
        if (_rgbImgsItr != _rgbImageFiles.end())
        {
            image = getRgbImage(*_rgbImgsItr);
            _rgbImgsItr++;
            return true;
        }
        else
        {
            return false;
        }
    }

    bool isCurrentDepthValid()
    {
        return _isCurrentDepthValid;
    }

  private:
    std::vector<std::string>           _depthImageFiles, _rgbImageFiles;
    std::vector<std::string>::iterator _depthImgsItr, _rgbImgsItr;
    bool                               _isCurrentDepthValid;
    std::vector<RoboticsPose>          camPoses_;
};
