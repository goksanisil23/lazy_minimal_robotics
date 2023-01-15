#include <chrono>
#include <memory>
#include <string>

#include "argparse.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include "OpticalFlowSparse.h"
#include "TimeUtil.h"

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("sparse_optical_flow");
    program.add_argument("--image_folder").help("path of the image folder");
    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::vector<cv::String> imagePaths;
    cv::glob(program.get<std::string>("--image_folder"), imagePaths, true);

    auto    imgPathItr = imagePaths.begin();
    cv::Mat image1     = cv::imread(*imgPathItr, 0);
    imgPathItr++;
    // ------------------ GAUSS-NEWTON OPTICAL FLOW ------------------ //
    std::unique_ptr<OpticalFlowSparse> oflow{std::make_unique<OpticalFlowSparse>()};
    while (imgPathItr != imagePaths.end())
    {
        cv::Mat image2 = cv::imread(*imgPathItr, 0);

        std::vector<cv::KeyPoint> kpsImg1Out;
        std::vector<cv::KeyPoint> kpsImg2Out;
        std::vector<bool>         isFlowOkOut;
        oflow->Step(image1, image2, kpsImg1Out, kpsImg2Out, isFlowOkOut);

        cv::destroyAllWindows();
        image1 = image2;
        imgPathItr++;
    }

    // ------------------ GAUSS-NEWTON OPTICAL FLOW ------------------ //
    // std::unique_ptr<OpticalFlow> oflow{std::make_unique<OpticalFlow>()};
    // std::vector<cv::KeyPoint>    kpsImg1Out;
    // std::vector<cv::KeyPoint>    kpsImg2Out;
    // std::vector<bool>            isFlowOkOut;
    // oflow->Step(image1, image2, kpsImg1Out, kpsImg2Out, isFlowOkOut);

    // // ------------------ OPENCV OPTICAL FLOW ------------------ //
    // std::vector<cv::KeyPoint> kpsImg1;
    // cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.1, 20.0);
    // detector->detect(image1, kpsImg1);

    // std::vector<bool>        success;
    // std::vector<cv::Point2f> pt1, pt2;
    // for (auto &kp : kpsImg1)
    //     pt1.push_back(kp.pt);
    // std::vector<uchar> status;
    // std::vector<float> error;
    // auto               t0 = time_util::chronoNow();
    // cv::calcOpticalFlowPyrLK(image1, image2, pt1, pt2, status, error);
    // auto t1 = time_util::chronoNow();
    // time_util::showTimeDuration(t1, t0, "opencv oflow: ");

    // cv::Mat image2Bgr;
    // cv::cvtColor(image2, image2Bgr, cv::COLOR_GRAY2BGR);
    // for (size_t i = 0; i < pt2.size(); i++)
    // {
    //     if (status[i])
    //     {
    //         cv::circle(image2Bgr, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
    //         cv::line(image2Bgr, kpsImg1[i].pt, pt2[i], cv::Scalar(0, 250, 0));
    //     }
    // }
    // cv::imshow("opencv", image2Bgr);
    // cv::waitKey(0);

    return 0;
}