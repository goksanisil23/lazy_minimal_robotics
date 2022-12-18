#include "Map.h"

#include <limits>

namespace landmarkSim2D
{

Map::Map(const std::string &mapFilePath)
{
    std::ifstream mapFile(mapFilePath);
    std::string   line;
    while (std::getline(mapFile, line))
    {
        std::istringstream istream(line);
        int                landmarkIdx;
        float              posX, posY;

        istream >> landmarkIdx >> posX >> posY;
        landmarks.push_back(Landmark(landmarkIdx, posX, posY));
    }
}

void Map::GenerateCircularLandmarks(const float &radius, const float &angleIntervalRad)
{
    std::uniform_real_distribution<float> uniformDist(-radius / 5.0,
                                                      radius / 5.0); // limit the randomness to 10% of the radius

    int16_t numLandmarks = 2 * M_PI / angleIntervalRad;

    for (int16_t lmIdx = 0; lmIdx < numLandmarks; lmIdx++)
    {
        float angleLm = lmIdx * angleIntervalRad;
        float xLm     = radius * cos(angleLm);
        float yLm     = radius * sin(angleLm);

        // Add some randomness to break the even pattern
        {
            xLm += uniformDist(randGenEngine_);
            yLm += uniformDist(randGenEngine_);
        }

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

    return landmarksInRange;
}

void Map::DumpMapToFile(const std::string &outputFileName)
{
    std::ofstream mapFile;
    mapFile.open(outputFileName);

    for (const auto &landmark : landmarks)
    {
        mapFile << landmark.id << " " << landmark.posX << " " << landmark.posY << std::endl;
    }

    mapFile.close();
}

Map::BboxExtent Map::GetBoundingExtentOfMap()
{
    float low_x  = std::numeric_limits<float>::max();
    float low_y  = std::numeric_limits<float>::max();
    float high_x = -std::numeric_limits<float>::max();
    float high_y = -std::numeric_limits<float>::max();

    for (const auto &landmark : landmarks)
    {
        if (landmark.posX < low_x)
            low_x = landmark.posX;
        if (landmark.posY < low_y)
            low_y = landmark.posY;
        if (landmark.posX > high_x)
            high_x = landmark.posX;
        if (landmark.posY > high_y)
            high_y = landmark.posY;
    }

    return Map::BboxExtent{low_x, low_y, high_x, high_y};
}

float Map::Landmark::Distance2(const Landmark &otherLandmark) const
{
    return (this->posX - otherLandmark.posX) * (this->posX - otherLandmark.posX) +
           (this->posY - otherLandmark.posY) * (this->posY - otherLandmark.posY);
}

} // namespace landmarkSim2D