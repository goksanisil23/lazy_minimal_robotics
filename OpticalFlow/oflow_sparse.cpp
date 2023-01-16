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

    // Create colors for the tracks
    std::vector<cv::Scalar> colors;
    cv::RNG                 rng;
    for (int i = 0; i < OpticalFlowSparse::MAX_DETECTOR_CORNERS; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(cv::Scalar(r, g, b));
    }

    auto    imgPathItr = imagePaths.begin();
    cv::Mat image1     = cv::imread(*imgPathItr);
    imgPathItr++;
    cv::Mat mask = cv::Mat::zeros(image1.size(), image1.type());
    // ------------------ GAUSS-NEWTON OPTICAL FLOW ------------------ //
    std::vector<cv::KeyPoint>          kpsImg1;
    std::unique_ptr<OpticalFlowSparse> oflow{std::make_unique<OpticalFlowSparse>()};
    oflow->Detect(image1, kpsImg1);

    while (imgPathItr != imagePaths.end())
    {
        cv::Mat image2 = cv::imread(*imgPathItr);

        std::vector<cv::KeyPoint> kpsImg2;
        std::vector<bool>         isFlowOk;
        oflow->Track(image1, image2, kpsImg1, kpsImg2, isFlowOk);

        std::vector<cv::KeyPoint> goodNew;
        for (uint i = 0; i < kpsImg1.size(); i++)
        {
            if (isFlowOk.at(i))
            {
                goodNew.push_back(kpsImg2.at(i));
                cv::line(mask, kpsImg2[i].pt, kpsImg1[i].pt, colors[i], 2);
                cv::circle(image2, kpsImg2[i].pt, 5, colors[i], -1);
            }
        }
        cv::Mat drawingImg;
        cv::add(image2, mask, drawingImg);
        cv::imshow("Frame", drawingImg);
        int keyboard = cv::waitKey(33);
        kpsImg1      = goodNew;

        image1 = image2.clone();
        imgPathItr++;
    }

    return 0;
}