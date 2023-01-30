#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

int main()
{
    // Create a blank image with a white background
    cv::Mat image(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));

    // Initial position and velocity of the ball
    cv::Point pos(0, 300);
    cv::Point vel(50, -50);

    // Time step for animation
    double dt = 0.1;

    // Gravitational acceleration
    cv::Point g(0, 9.8);

    // Random number generator for velocity noise
    std::random_device         rd;
    std::mt19937               gen(rd());
    std::normal_distribution<> noise(0, 2);

    // Loop to simulate the motion of the ball
    while (1)
    {
        // Clear the image
        image = cv::Scalar(255, 255, 255);

        // Add noise to the velocity
        // vel.x += noise(gen);
        // vel.y += noise(gen);

        // Draw the ball at its current position
        cv::circle(image, pos, 10, cv::Scalar(0, 0, 255), -1);

        // Update the velocity and position of the ball
        vel += g * dt;
        pos += vel * dt;

        // Check if the ball hits the ground
        if (pos.y >= image.rows || pos.x >= image.cols || pos.x <= 0 || pos.y <= 0)
        {
            break;
        }

        // Show the image
        cv::imshow("Projectile Motion", image);

        // Wait for a while before updating the animation
        cv::waitKey(50);
    }

    return 0;
}