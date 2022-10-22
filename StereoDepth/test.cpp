#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/opencv.hpp>

namespace my_util
{
constexpr auto &chronoNow = std::chrono::high_resolution_clock::now;
using time_point          = decltype(std::chrono::high_resolution_clock::now());
} // namespace my_util

void showTimeDuration(const my_util::time_point &t2, const my_util::time_point &t1, const std::string &message)
{
    std::cout << message << std::chrono::duration<double, std::chrono::seconds::period>(t2 - t1).count() << std::endl;
}

typedef cv::Point3_<uint8_t> Pixel;
using namespace std::chrono_literals;

void someThreshold(Pixel &pixel)
{
    if (pow(double(pixel.x) / 10, 2.5) > 100)
    {
        pixel.x = 255;
        pixel.y = 255;
        pixel.z = 255;
    }
    else
    {
        pixel.x = 0;
        pixel.y = 0;
        pixel.z = 0;
    }
}

void naiveLoop(cv::Mat image)
{
    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            // Pixel pixel = image.at<Pixel>(r, c);
            // someThreshold(pixel);
            // image.at<Pixel>(r, c) = pixel;
            someThreshold(image.at<Pixel>(r, c));
        }
    }
}

void pointerLoop(cv::Mat image)
{
    // Pointer to the 1st pixel
    Pixel *pixelPtr = image.ptr<Pixel>(0, 0);
    // cv::Mat objects created using create() method are stored in 1 contiguous memory block
    const Pixel *endPixelPtr = pixelPtr + image.cols * image.rows;
    for (; pixelPtr != endPixelPtr; pixelPtr++)
    {
        someThreshold(*pixelPtr);
    }
}

void forEachLoop(cv::Mat image)
{
    image.forEach<Pixel>([](Pixel &pixel, const int *position) { someThreshold(pixel); });
}

int main()
{
    cv::Mat image = cv::imread("/home/goksan/Downloads/4k.jpg");
    std::cout << image.rows << " " << image.cols << std::endl;

    auto t0 = my_util::chronoNow();
    while (true)
        naiveLoop(image);
    auto t1 = my_util::chronoNow();
    pointerLoop(image);
    auto t2 = my_util::chronoNow();
    forEachLoop(image);
    auto t3 = my_util::chronoNow();

    showTimeDuration(t1, t0, "naive: ");
    showTimeDuration(t2, t1, "ptr  : ");
    showTimeDuration(t3, t2, "for_e: ");

    return 0;
}