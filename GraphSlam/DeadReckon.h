#pragma once

#include "raylib-cpp.hpp"

struct DeadReckon
{
    raylib::Vector2 position;
    float           heading;

    void update(const raylib::Vector2 delta_pos, const float delta_rot);
};