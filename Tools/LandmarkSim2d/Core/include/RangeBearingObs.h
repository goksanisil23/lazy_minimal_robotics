#pragma once

#include <cstdint>

namespace landmarkSim2D
{

struct RangeBearingObs
{
    RangeBearingObs() = default;

    explicit RangeBearingObs(const int16_t &id, const float &range, const float &angleRad)
        : id{id}, range{range}, angleRad{angleRad}
    {
    }

    int16_t id;
    float   range;
    float   angleRad;
};

} // namespace landmarkSim2D
