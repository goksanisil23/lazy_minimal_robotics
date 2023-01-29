#pragma once

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/iteration_callback.h>
#include <ceres/loss_function.h>
#include <ceres/types.h>

#include "KalmanTuner.h"

struct KalmanErrorTerm
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KalmanErrorTerm(const std::vector<Eigen::Vector2d>     &noisyCtrlVec,
                    const std::vector<Eigen::Vector2d>     &noisyMeasVec,
                    const std::vector<Eigen::Vector2d>     &trueMeasVec,
                    const Eigen::Vector<double, STATESIZE> &initState);

    template <typename T>
    bool operator()(const T *const filterNoiseParamsPtr, T *residualsPtr) const;

    static ceres::CostFunction *Create(const std::vector<Eigen::Vector2d>     &noisyCtrlVec,
                                       const std::vector<Eigen::Vector2d>     &noisyMeasVec,
                                       const std::vector<Eigen::Vector2d>     &trueMeasVec,
                                       const Eigen::Vector<double, STATESIZE> &initState);

    template <typename T>
    Eigen::Vector<T, STATESIZE> MotionModel(const Eigen::Vector<T, STATESIZE>     &X,
                                            const Eigen::Vector<double, CTRLSIZE> &U) const;

    template <typename T>
    Eigen::Vector<T, MEASSIZE> ObservationModel(const Eigen::Vector<T, STATESIZE> &X) const;

    template <typename T>
    Eigen::Vector<T, STATESIZE> Predict(Eigen::Vector<T, STATESIZE>                  &xhat,
                                        Eigen::Matrix<T, STATESIZE, STATESIZE>       &P,
                                        const Eigen::Vector<double, CTRLSIZE>        &U,
                                        const Eigen::Matrix<T, STATESIZE, STATESIZE> &Q) const;

    template <typename T>
    Eigen::Vector<T, STATESIZE> Correct(Eigen::Vector<T, STATESIZE>                &xhat,
                                        Eigen::Matrix<T, STATESIZE, STATESIZE>     &P,
                                        const Eigen::Vector<double, MEASSIZE>      &Z,
                                        const Eigen::Matrix<T, MEASSIZE, MEASSIZE> &R) const;

    template <typename T>
    Eigen::Matrix<T, STATESIZE, STATESIZE> JacobianF(const Eigen::Vector<T, STATESIZE>     &X_hat_prio,
                                                     const Eigen::Vector<double, CTRLSIZE> &U) const;

    Eigen::Matrix<double, MEASSIZE, STATESIZE> JacobianH() const;

  private:
    // member variables
    std::vector<Eigen::Vector<double, CTRLSIZE>> noisyCtrlVec_;
    std::vector<Eigen::Vector<double, MEASSIZE>> noisyMeasVec_;
    std::vector<Eigen::Vector<double, MEASSIZE>> trueMeasVec_;

    Eigen::Vector<double, STATESIZE> initState_;
};