#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <vector>

#include "Pose2D.h"

namespace landmarkSim2D
{

class Map
{
  public:
    struct Landmark
    {
        Landmark() = default;
        explicit Landmark(const int16_t &id, const float &positionX, const float &positionY)
            : id(id), posX(positionX), posY(positionY)
        {
        }

        float Distance2(const Landmark &otherLandmark) const;

        int16_t id;
        float   posX;
        float   posY;
    };

    struct BboxExtent
    {
        explicit BboxExtent(const float &lowX, const float &lowY, const float &highX, const float &highY)
            : lowX{lowX}, lowY{lowY}, highX{highX}, highY{highY}
        {
        }

        float lowX, lowY, highX, highY;
    };

    // Methods
    Map() = default;
    Map(const std::string &mapFilePath); // construct the map from file contents
    void                  GenerateCircularLandmarks(const float &radius, const float &angleIntervalRad);
    std::vector<Landmark> GetLandmarksWithinRadius(const Pose2D &pose, const float &radius);
    BboxExtent            GetBoundingExtentOfMap();
    void                  DumpMapToFile(const std::string &outputFileName);

    // Variables
    std::vector<Landmark> landmarks;
};
} // namespace landmarkSim2D