#pragma once

#include "raylib-cpp.hpp"

struct Landmark
{
    static constexpr float kRadius{5.F};

    raylib::Vector2 position_;
    int             id_;
};

struct LandmarkMeasurement
{
    float range;
    float bearing;
    int   id;
};