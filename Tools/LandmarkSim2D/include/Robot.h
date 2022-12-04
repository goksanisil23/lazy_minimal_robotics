#pragma once

#include <cstdint>
#include <math.h>
#include <random>
#include <vector>

#include "Map.h"
#include "Pose2D.h"
#include "RangeBearingObs.h"

namespace landmarkSim2D
{

constexpr float SIGMA_LANDMARK_RANGE   = 0.2f; // landmark measurement range uncertainty (m.)
constexpr float SIGMA_LANDMARK_BEARING = 1.0f; // landmark measurement bearing uncertainty (deg.)

struct ControlInput
{
    ControlInput() = default;

    explicit ControlInput(const float &linearVel, const float &angularVel)
        : linearVel(linearVel), angularVel(angularVel)
    {
    }

    float linearVel;
    float angularVel;
};

class Robot
{
  public:
    ControlInput controlInput;

    Robot(const Pose2D &initPose, const Map &map);

    void Act(const float &dt);
    void Sense();

    void MoveWithControlInput(const float &dt);
    void GenerateLandmarkObservations();
    void DetectLandmarksWithNoise();

    float GetSensorRange() const;

    Pose2D                       pose;
    std::vector<RangeBearingObs> observedLandmarks;

  private:
    Map   map_;
    float sensorRange_;

    // Normal distributions
    std::default_random_engine      randGen_;
    std::normal_distribution<float> noiseRange_;
    std::normal_distribution<float> noiseBearing_;
};

} // namespace landmarkSim2D