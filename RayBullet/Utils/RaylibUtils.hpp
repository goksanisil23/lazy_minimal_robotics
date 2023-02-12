#pragma once

#include "raylib-cpp.hpp"
#include <iostream>

inline std::ostream &operator<<(std::ostream &os, const raylib::Vector3 &vec)
{
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")" << std::endl;
    return os;
}