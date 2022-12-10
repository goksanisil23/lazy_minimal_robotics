#pragma once

namespace landmarkSim2D
{
struct Pose2D
{
    Pose2D() = default;

    explicit Pose2D(const float &posX, const float &posY, const float &yawRad) : posX(posX), posY(posY), yawRad(yawRad)
    {
    }

    float posX;
    float posY;
    float yawRad;
};
} // namespace landmarkSim2D