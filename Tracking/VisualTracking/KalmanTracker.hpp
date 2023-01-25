#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

class KalmanTracker
{
  public:
    KalmanTracker(const Eigen::Vector<double, 6> &initState, const std::vector<double> &tunedProcessNoiseParams)
        : initState_{initState}
    {
        stateTransMtx_       = Eigen::MatrixXd::Identity(stateSize_, stateSize_);
        stateTransMtx_(0, 2) = dt;
        stateTransMtx_(1, 3) = dt;

        measMtx_       = Eigen::MatrixXd::Zero(measSize_, stateSize_);
        measMtx_(0, 0) = 1.0;
        measMtx_(1, 1) = 1.0;
        measMtx_(2, 4) = 1.0;
        measMtx_(3, 5) = 1.0;

        measNoiseMtx_ = Eigen::MatrixXd::Identity(measSize_, measSize_) * 1e-7;

        x_hat_ = initState_;

        // Diagonals will be populated by tuned vars
        processNoiseMtx_       = Eigen::MatrixXd::Zero(stateSize_, stateSize_);
        processNoiseMtx_(0, 0) = tunedProcessNoiseParams.at(0);
        processNoiseMtx_(1, 1) = tunedProcessNoiseParams.at(1);
        processNoiseMtx_(2, 2) = tunedProcessNoiseParams.at(2);
        processNoiseMtx_(3, 3) = tunedProcessNoiseParams.at(3);
        processNoiseMtx_(4, 4) = tunedProcessNoiseParams.at(4);
        processNoiseMtx_(5, 5) = tunedProcessNoiseParams.at(5);

        stateErrorCovMtx_ = Eigen::MatrixXd::Identity(stateSize_, stateSize_);
    }

    Eigen::Vector<double, 6> Predict()
    {
        x_hat_            = stateTransMtx_ * x_hat_;
        stateErrorCovMtx_ = (stateTransMtx_ * stateErrorCovMtx_ * stateTransMtx_.transpose()) + processNoiseMtx_;
        return x_hat_;
    }

    Eigen::Vector<double, 6> Correct(const Eigen::Vector<double, 4> &measBbox)
    {
        Eigen::MatrixXd K = stateErrorCovMtx_ * measMtx_.transpose() *
                            (measMtx_ * stateErrorCovMtx_ * measMtx_.transpose() + measNoiseMtx_).inverse();

        x_hat_            = x_hat_ + K * measBbox - K * measMtx_ * x_hat_;
        stateErrorCovMtx_ = stateErrorCovMtx_ - K * measMtx_ * stateErrorCovMtx_;
        return x_hat_;
    }

  private:
    Eigen::Vector<double, 6> initState_;
    Eigen::Vector<double, 6> x_hat_;

    const int32_t stateSize_{6}; // c_x,c_y,v_x,v_y,w,h (center of bbox, velocity of bbox, height, width)
    const int32_t measSize_{4};  // c_x, c_y, w, h
    const double  dt{0.05};

    Eigen::MatrixXd stateTransMtx_;    // n*n
    Eigen::MatrixXd measMtx_;          // m*n
    Eigen::MatrixXd measNoiseMtx_;     // m*m
    Eigen::MatrixXd processNoiseMtx_;  // n*n
    Eigen::MatrixXd stateErrorCovMtx_; // n*n
};