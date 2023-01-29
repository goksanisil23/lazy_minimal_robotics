#pragma once

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/iteration_callback.h>
#include <ceres/loss_function.h>
#include <ceres/types.h>

constexpr int STATESIZE = 4;
constexpr int MEASSIZE  = 2;
constexpr int CTRLSIZE  = 2;

constexpr int OPT_PARAMS_SIZE = 6;

constexpr double INIT_OPT_PARAMS_VAL = 0.1;

constexpr double DT = 0.1;

class KalmanTuner : public std::enable_shared_from_this<KalmanTuner>
{
  public:
    KalmanTuner();

    void Tune(const std::vector<Eigen::Vector2d> &noisyCtrlVec,
              const std::vector<Eigen::Vector2d> &noisyMeasVec,
              const std::vector<Eigen::Vector2d> &trueMeasVec);

    // system dimensions
    static const int32_t stateSize_{STATESIZE}; // x,y,h,v
    static const int32_t measSize_{MEASSIZE};   // x,y
    static const int32_t ctrlSize{CTRLSIZE};    // v,w

    std::vector<double> filterNoiseParams; // [R,Q]
};