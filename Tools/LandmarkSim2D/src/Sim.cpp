#include "Sim.h"

#include <memory>
#include <tuple>

namespace landmarkSim2D
{
Sim::Sim()
{
    // Create the map with 1 big and 1 small circular landmarks
    float smallCircleRadius             = 10.0;
    float bigCircleRadius               = 15.0;
    float smallCircleLmAngleIntervalDeg = 3.0;
    float bigCircleLmAngleIntervalDeg   = 5.0;

    map = std::make_shared<Map>();
    map->GenerateCircularLandmarks(smallCircleRadius, smallCircleLmAngleIntervalDeg * M_PI / 180.0);
    map->GenerateCircularLandmarks(bigCircleRadius, bigCircleLmAngleIntervalDeg * M_PI / 180.0);

    Pose2D robotInitPose((smallCircleRadius + bigCircleRadius) / 2.0, 0.0, M_PI / 2);
    robot = std::make_shared<Robot>(robotInitPose, *map);
}

void Sim::Init()
{
}

void Sim::Step(const float &dt)
{
    robot->Act(dt);
}

} // namespace landmarkSim2D