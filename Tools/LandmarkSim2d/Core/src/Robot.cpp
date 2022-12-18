#include "Robot.h"

#include <iostream>

namespace landmarkSim2D
{

Robot::Robot(const Pose2D &initPose, const Map &map) : truePose_(initPose), map_(map), sensorRange_{SENSOR_RANGE}
{
    drPose_ = truePose_;

    noiseRange_      = std::normal_distribution<float>{0.0f, SIGMA_LANDMARK_RANGE};
    noiseBearing_    = std::normal_distribution<float>{0.0f, SIGMA_LANDMARK_BEARING * M_PI / 180.0};
    noiseCtrlLinVel_ = std::normal_distribution<float>{0.0f, SIGMA_CONTROL_LIN_VEL};
    noiseCtrlAngVel_ = std::normal_distribution<float>{0.0f, SIGMA_CONTROL_ANG_VEL * M_PI / 180.0};
}

ControlInput Robot::GenerateCircularControlInput()
{
    ControlInput ctrlInput;
    ctrlInput.angularVel = 0.1;
    ctrlInput.linearVel  = trueControlInput_.angularVel * (25.0 / 2.0); // v = w*R
    return ctrlInput;
}

void Robot::Act(const float &dt)
{
    // Generate control input
    trueControlInput_ = ControlInput{GenerateCircularControlInput()};

    measuredControlInput_ = MeasureControlInputWithNoise();

    truePose_ = IterateMotionModel(truePose_, dt, trueControlInput_);
    drPose_   = IterateMotionModel(drPose_, dt, measuredControlInput_);
}

Pose2D Robot::IterateMotionModel(const Pose2D &prevPose, const float &dt, const ControlInput &controlInput)
{
    Pose2D newPose;
    newPose.posX   = prevPose.posX + controlInput.linearVel * cos(prevPose.yawRad) * dt;
    newPose.posY   = prevPose.posY + controlInput.linearVel * sin(prevPose.yawRad) * dt;
    newPose.yawRad = prevPose.yawRad + controlInput.angularVel * dt;

    return newPose;
}

// Get all the landmarks within range from the map and detect with some noise
void Robot::DetectLandmarksWithNoise()
{
    observedLandmarks_.clear();
    // This returns ideal detections
    auto landmarksInRange(map_.GetLandmarksWithinRadius(this->truePose_, this->sensorRange_));
    // Generate bearing angle
    int16_t obsId = 0;
    for (const auto &landmarkInRange : landmarksInRange)
    {
        RangeBearingObs observation;
        float           angleToLandmark =
            atan2(landmarkInRange.posY - truePose_.posY, landmarkInRange.posX - truePose_.posX) - truePose_.yawRad;
        observation.angleRad = angleToLandmark;
        // observation.angleRad += noiseBearing_(randGen_); // add noise
        observation.range = sqrt((landmarkInRange.posY - truePose_.posY) * (landmarkInRange.posY - truePose_.posY) +
                                 (landmarkInRange.posX - truePose_.posX) * (landmarkInRange.posX - truePose_.posX));
        // observation.range += noiseRange_(randGen_); // add noise
        observation.id = obsId;

        observedLandmarks_.push_back(observation);
        obsId++;
    }
}

ControlInput Robot::MeasureControlInputWithNoise()
{
    ControlInput noisyControlInputMeas;

    noisyControlInputMeas.linearVel  = trueControlInput_.linearVel + noiseCtrlLinVel_(randGen_);
    noisyControlInputMeas.angularVel = trueControlInput_.angularVel + noiseCtrlAngVel_(randGen_);

    return noisyControlInputMeas;
}

void Robot::Sense()
{
    DetectLandmarksWithNoise();
}

float Robot::GetSensorRange()
{
    return SENSOR_RANGE;
}

} // namespace landmarkSim2D