#include "Robot.h"

#include <iostream>

namespace landmarkSim2D
{
Robot::Robot(const Pose2D &initPose, const Map &map) : pose(initPose), map_(map)
{
    sensorRange_ = 3.0f;

    noiseRange_   = std::normal_distribution<float>{0.0f, SIGMA_LANDMARK_RANGE};
    noiseBearing_ = std::normal_distribution<float>{0.0f, SIGMA_LANDMARK_BEARING * M_PI / 180.0};
}

void Robot::Act(const float &dt)
{
    // Generate control input
    controlInput.angularVel = 0.1;
    controlInput.linearVel  = controlInput.angularVel * (25.0 / 2.0); // v = w*R
    MoveWithControlInput(dt);
}

void Robot::MoveWithControlInput(const float &dt)
{
    Pose2D prevPose(pose);

    pose.posX += controlInput.linearVel * cos(pose.yawRad) * dt;
    pose.posY += controlInput.linearVel * sin(pose.yawRad) * dt;
    pose.yawRad += controlInput.angularVel * dt;
}

// Get all the landmarks within range from the map and detect with some noise
void Robot::DetectLandmarksWithNoise()
{
    observedLandmarks.clear();
    // This returns ideal detections
    auto landmarksInRange(map_.GetLandmarksWithinRadius(this->pose, this->sensorRange_));
    // Generate bearing angle
    int16_t obsId = 0;
    for (const auto &landmarkInRange : landmarksInRange)
    {
        RangeBearingObs observation;
        float angleToLandmark = atan2(landmarkInRange.posY - pose.posY, landmarkInRange.posX - pose.posX) - pose.yawRad;
        observation.angleRad  = angleToLandmark + noiseBearing_(randGen_);
        observation.range     = sqrt((landmarkInRange.posY - pose.posY) * (landmarkInRange.posY - pose.posY) +
                                 (landmarkInRange.posX - pose.posX) * (landmarkInRange.posX - pose.posX));
        observation.range += noiseRange_(randGen_);
        observation.id = obsId;

        observedLandmarks.push_back(observation);
        obsId++;
    }
}

void Robot::Sense()
{
    DetectLandmarksWithNoise();
}

float Robot::GetSensorRange() const
{
    return sensorRange_;
}

} // namespace landmarkSim2D