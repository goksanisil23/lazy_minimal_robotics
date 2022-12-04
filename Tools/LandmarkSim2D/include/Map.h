#pragma once

#include <cstdint>
#include <math.h>
#include <vector>

#include "Pose2D.h"

namespace landmarkSim2D
{
class Map
{
  public:
    struct Landmark
    {
        explicit Landmark(const int16_t &id, const float &positionX, const float &positionY)
            : id(id), posX(positionX), posY(positionY)
        {
        }

        int16_t id;
        float   posX;
        float   posY;
    };

    // Methods
    Map() = default;
    void                  GenerateCircularLandmarks(const float &radius, const float &angleIntervalRad);
    std::vector<Landmark> GetLandmarksWithinRadius(const Pose2D &pose, const float &radius);

    // Variables
    std::vector<Landmark> landmarks;
};
} // namespace landmarkSim2D