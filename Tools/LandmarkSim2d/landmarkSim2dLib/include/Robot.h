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
constexpr float SIGMA_CONTROL_LIN_VEL  = 1.0f; // control input measurement uncertainty (m/s)
constexpr float SIGMA_CONTROL_ANG_VEL  = 1.0f; // control input measurement uncertainty (deg/s)

constexpr float SENSOR_RANGE = 3.0f;

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
    ControlInput trueControlInput_;
    ControlInput measuredControlInput_;

    Robot(const Pose2D &initPose, const Map &map);

    void Act(const float &dt);
    void Sense();

    static Pose2D IterateMotionModel(const Pose2D &prevPose, const float &dt, const ControlInput &controlInput);

    void         GenerateLandmarkObservations();
    void         DetectLandmarksWithNoise();
    ControlInput MeasureControlInputWithNoise();
    static float GetSensorRange();

    Pose2D truePose_;
    Pose2D drPose_; // deadreckon pose

    std::vector<RangeBearingObs> observedLandmarks_;

  private:
    Map   map_;
    float sensorRange_;

    // Normal distributions
    std::default_random_engine      randGen_;
    std::normal_distribution<float> noiseRange_;
    std::normal_distribution<float> noiseBearing_;
    std::normal_distribution<float> noiseCtrlLinVel_;
    std::normal_distribution<float> noiseCtrlAngVel_;
};

} // namespace landmarkSim2D