#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/iteration_callback.h>
#include <ceres/loss_function.h>
#include <ceres/types.h>

#include "KalmanTuner.h"

struct KalmanErrorTerm
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KalmanErrorTerm(const std::vector<Eigen::Vector<double, 4>> &gtBboxVec, const Eigen::Vector<double, 6> &initState);

    template <typename T>
    bool operator()(const T *const filterNoiseParamsPtr, T *residualsPtr) const;

    static ceres::CostFunction *Create(const std::vector<Eigen::Vector<double, 4>> &gtBboxVec,
                                       const Eigen::Vector<double, 6>              &initState);

    // c_x,c_y,w,h
    template <typename T>
    T IoU(const Eigen::Vector<double, 4> &gtBboxMes, const Eigen::Vector<T, 6> &predState) const;

  private:
    // member variables
    std::vector<Eigen::Vector<double, 4>> gtBboxVec_;
    Eigen::Vector<double, 6>              initState_;
};