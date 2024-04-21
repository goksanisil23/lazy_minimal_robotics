#pragma once

#include "raylib-cpp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "Landmark.h"

class Agent
{
  public:
    static constexpr float kRadius{10.F};
    static constexpr float kSensorRange{100.F};

    Agent(const raylib::Vector2 position_initial, const float heading_initial);

    raylib::Vector2 position_;
    float           heading_{0.0}; // heading angle_ in radians

    std::vector<LandmarkMeasurement> landmark_measurements_{};

    std::default_random_engine      noise_generator_;
    std::normal_distribution<float> linear_vel_noise_dist_{0.1, 1.0};   // Linear velocity noise
    std::normal_distribution<float> angular_vel_noise_dist_{0.04, 0.2}; // Angular velocity noise
    // std::normal_distribution<float> linear_vel_noise_dist_{0.0, 0};         // Linear velocity noise
    // std::normal_distribution<float> angular_vel_noise_dist_{0.0, 0};        // Angular velocity noise
    std::normal_distribution<float> landmark_range_noise_dist_{0.0, 0.0};   // Sensor noise for distance
    std::normal_distribution<float> landmark_bearing_noise_dist_{0.0, 0.0}; // Sensor noise for bearing

    // IMU simulated measurements
    std::pair<float, float> readIMU(float linearVelocity, float angularVelocity);

    // Update movement based on actual commands without noise
    void updateMovement(float linearVelocity, float angularVelocity);

    // Sensing landmarks
    void senseLandmarks(const std::vector<Landmark> &landmarks);
};