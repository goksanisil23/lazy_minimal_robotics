#pragma once

#include <cstdint>

namespace landmarkSim2D
{

struct RangeBearingObs
{
    int16_t id;
    float   range;
    float   angleRad;
};

} // namespace landmarkSim2D
