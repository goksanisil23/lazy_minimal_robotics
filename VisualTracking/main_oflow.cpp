#include <chrono>
#include <memory>
#include <string>

// #include <opencv4/opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include "OpticalFlow.hpp"
// #include "TimeUtil.h"

int main(int argc, char **argv)
{
    std::string file1  = "/home/goksan/Downloads/slambook2/ch8/LK1.png"; // first image
    std::string file2  = "/home/goksan/Downloads/slambook2/ch8/LK2.png"; // second image
    cv::Mat     image1 = cv::imread(file1, 0);
    cv::Mat     image2 = cv::imread(file2, 0);

    // ------------------ GAUSS-NEWTON OPTICAL FLOW ------------------ //
    std::unique_ptr<OpticalFlow> oflow{std::make_unique<OpticalFlow>()};
    oflow->Step(image1, image2);

    // ------------------ OPENCV OPTICAL FLOW ------------------ //
    std::vector<cv::KeyPoint> kpsImg1;
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20.0);
    detector->detect(image1, kpsImg1);

    std::vector<bool>        success;
    std::vector<cv::Point2f> pt1, pt2;
    for (auto &kp : kpsImg1)
        pt1.push_back(kp.pt);
    std::vector<uchar> status;
    std::vector<float> error;
    // auto               t0 = time_util::chronoNow();
    cv::calcOpticalFlowPyrLK(image1, image2, pt1, pt2, status, error);
    // auto t1 = time_util::chronoNow();
    // time_util::showTimeDuration(t1, t0, "opencv oflow: ");

    cv::Mat image2Bgr;
    cv::cvtColor(image2, image2Bgr, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < pt2.size(); i++)
    {
        if (status[i])
        {
            cv::circle(image2Bgr, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(image2Bgr, kpsImg1[i].pt, pt2[i], cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("opencv", image2Bgr);
    cv::waitKey(0);

    return 0;
}