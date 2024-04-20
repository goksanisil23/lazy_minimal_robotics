#pragma once

#include "raylib-cpp.hpp"

class Odometry
{
  public:
    raylib::Vector2 delta_position_;
    float           delta_heading_;

    void update(const float &linear_vel_meas, const float &angular_vel_meas);
};