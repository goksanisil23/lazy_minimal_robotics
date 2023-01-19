#include <chrono>
#include <memory>
#include <string>

#include "argparse.hpp"
#include <opencv2/opencv.hpp>

#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("dense_optical_flow");
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

    cv::cuda::GpuMat prev_frame, next_frame, flow_gpu;
    // create an instance of the Farneback optical flow algorithm
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farn = cv::cuda::FarnebackOpticalFlow::create();

    auto    imgPathItr = imagePaths.begin();
    cv::Mat image1     = cv::imread(*imgPathItr, 0);

    while (imgPathItr != imagePaths.end())
    {
        cv::Mat image2 = cv::imread(*imgPathItr, 0);

        prev_frame.upload(image1);
        next_frame.upload(image2);
        farn->calc(prev_frame, next_frame, flow_gpu);

        // download the flow field to the host
        cv::Mat flow;
        flow_gpu.download(flow);

        // cv::Mat flow(image2.size(), CV_32FC2);
        // cv::calcOpticalFlowFarneback(image1, image2, flow, 0.5, 3, 15, 10, 5, 1.1, 0);

        // Visualization
        cv::Mat flowParts[2];
        cv::split(flow, flowParts);
        cv::Mat magnitude, angle, magnitudeNormalized;
        cv::cartToPolar(flowParts[0], flowParts[1], magnitude, angle, true);
        cv::normalize(magnitude, magnitudeNormalized, 0.0, 1.0, cv::NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));

        //build hsv image
        cv::Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magnitudeNormalized;
        cv::merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

        cv::imshow("flow", bgr);
        cv::waitKey(0);

        image1 = image2.clone();
        imgPathItr++;
    }

    return 0;
}