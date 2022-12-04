#include "Map.h"

namespace landmarkSim2D
{
void Map::GenerateCircularLandmarks(const float &radius, const float &angleIntervalRad)
{
    int16_t numLandmarks = 2 * M_PI / angleIntervalRad;

    for (int16_t lmIdx = 0; lmIdx < numLandmarks; lmIdx++)
    {
        float angleLm = lmIdx * angleIntervalRad;
        float xLm     = radius * cos(angleLm);
        float yLm     = radius * sin(angleLm);

        Landmark lm(lmIdx, xLm, yLm);
        landmarks.push_back(lm);
    }
}

std::vector<Map::Landmark> Map::GetLandmarksWithinRadius(const Pose2D &pose, const float &radius)
{
    std::vector<Map::Landmark> landmarksInRange;
    // Find the landmarks in the map that are within robot's sensor range
    for (auto landmark : landmarks)
    {
        float distanceToLandmark = sqrt((landmark.posX - pose.posX) * (landmark.posX - pose.posX) +
                                        (landmark.posY - pose.posY) * (landmark.posY - pose.posY));

        if (distanceToLandmark < radius)
        {
            landmarksInRange.push_back(landmark);
        }
    }

    return std::move(landmarksInRange);
}

} // namespace landmarkSim2D