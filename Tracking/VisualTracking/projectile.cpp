#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

// Constants
constexpr double VEL_X{40.0};
constexpr double VEL_Y{-30.0};
constexpr double DT{0.05};
constexpr double CIRCLE_INIT_RAD{10.0};

// Helper funcs
std::string zeroPadString(const std::string &oldString)
{
    static constexpr size_t MAX_LEN{8};
    return std::string(MAX_LEN - std::min(MAX_LEN, oldString.length()), '0') + oldString;
}

void saveImageAndBbox(const cv::Mat &img, const cv::Rect &bbox)
{
    static size_t        frameCtr = 0;
    static std::string   folderPath{"/home/goksan/Downloads/tracking_datasets/lasot/simulated_ball/img/"};
    static std::ofstream gtFile{"/home/goksan/Downloads/tracking_datasets/lasot/simulated_ball/groundtruth.txt"};

    cv::imwrite(folderPath + zeroPadString(std::to_string(frameCtr)) + ".jpg", img);

    gtFile << bbox.tl().x << "," << bbox.tl().y << "," << bbox.width << "," << bbox.height << std::endl;

    frameCtr++;
}

cv::Rect2d getCircleBbox(const cv::Point2d &circlePos, const double &circleRad)
{
    cv::Point2d topLeft(circlePos - cv::Point2d(circleRad, circleRad) * 1.5);
    cv::Point2d botRight(circlePos + cv::Point2d(circleRad, circleRad) * 1.5);
    return cv::Rect2d(topLeft, botRight);
}

int main()
{
    // Create a blank image with a white background
    cv::Mat image(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));

    // Initial position and velocity of the ball
    cv::Point2d circlePos(CIRCLE_INIT_RAD, image.rows - CIRCLE_INIT_RAD);
    cv::Point2d circleVel(VEL_X, VEL_Y);

    // Time step for animation
    double dt = DT;

    // Gravitational acceleration
    // cv::Point g(0, 9.8);
    cv::Point2d g(0, 0);

    // Random number generator for velocity noise
    std::random_device         rd;
    std::mt19937               gen(rd());
    std::normal_distribution<> noise(0, 2);
    double                     circleRad{CIRCLE_INIT_RAD};
    cv::Rect                   bbox;

    // Loop to simulate the motion of the ball
    while (1)
    {
        // Clear the image
        image = cv::Scalar(255, 255, 255);

        // Add noise to the velocity
        // circleVel.x += noise(gen);
        // circleVel.y += noise(gen);

        // Draw the ball at its current position
        cv::circle(image, circlePos, circleRad, cv::Scalar(0, 0, 255), -1);
        cv::Mat drawImg;
        image.copyTo(drawImg);
        bbox = getCircleBbox(circlePos, circleRad);
        cv::rectangle(drawImg, bbox, cv::Scalar(250, 0, 0));
        saveImageAndBbox(image, bbox);

        // Update the circleVelocity and circlePosition of the ball
        circleVel += g * dt;
        circlePos += circleVel * dt;

        // Check if the ball hits the ground
        if (circlePos.y >= image.rows || circlePos.x >= image.cols || circlePos.x <= 0 || circlePos.y <= 0)
        {
            break;
        }

        // Show the image
        cv::imshow("Projectile Motion", image);

        // Wait for a while before updating the animation
        cv::waitKey(static_cast<int>(1.0 / DT));
    }

    return 0;
}