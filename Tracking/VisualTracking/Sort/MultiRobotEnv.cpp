#include "MultiRobotEnv.h"

MultiRobotEnv::Robot::Robot(const raylib::Vector2 &pos0, const float &radius) : position{pos0}, radius{radius}
{
}

void MultiRobotEnv::Robot::Draw() const
{
    DrawCircle(position.x, position.y, radius, BLUE);
}

MultiRobotEnv::MultiRobotEnv() : simWindow_(screenWidth, screenHeight, "Hungarian Assignment")
{
    SetTargetFPS(FPS);
}

void MultiRobotEnv::GenerateRobots(const size_t &areaWidth, const size_t &areaHeight, const size_t &numRobots)
{
    for (size_t robotIdx = 0; robotIdx < numRobots; robotIdx++)
    {
        float xCoord = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (areaWidth));
        float yCoord = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (areaHeight));
        robots_.emplace_back(Robot(raylib::Vector2(xCoord, yCoord), ROBOT_RADIUS));
    }
}

void MultiRobotEnv::DrawRobots()
{
    std::for_each(robots_.begin(), robots_.end(), [](const MultiRobotEnv::Robot &robot) { robot.Draw(); });
}

void MultiRobotEnv::MoveRobots()
{
    static auto prevTime = time_util::chronoNow();
    auto        currTime = time_util::chronoNow();
    double      dt       = time_util::getTimeDuration(currTime, prevTime);

    std::for_each(robots_.begin(),
                  robots_.end(),
                  [&dt](MultiRobotEnv::Robot &robot) { robot.position += raylib::Vector2(VX, VY) * dt; });

    prevTime = currTime;
}