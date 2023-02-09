#pragma once

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

class KalmanFilter
{
  public:
    KalmanFilter(const int32_t &stateSize, const int32_t &measSize, const Eigen::VectorXd &initState)
        : x_hat_{initState}, stateSize_{stateSize}, measSize_{measSize}
    {
    }

    Eigen::VectorXd Predict(const double &dt)
    {
        stateTransMtx_(0, 2) = dt;
        stateTransMtx_(1, 3) = dt;

        x_hat_            = stateTransMtx_ * x_hat_;
        stateErrorCovMtx_ = (stateTransMtx_ * stateErrorCovMtx_ * stateTransMtx_.transpose()) + processNoiseMtx_;
        return x_hat_;
    }

    Eigen::VectorXd Correct(const Eigen::Vector<double, 4> &meas)
    {
        Eigen::MatrixXd K = stateErrorCovMtx_ * measMtx_.transpose() *
                            (measMtx_ * stateErrorCovMtx_ * measMtx_.transpose() + measNoiseMtx_).inverse();

        x_hat_            = x_hat_ + K * (meas - measMtx_ * x_hat_);
        stateErrorCovMtx_ = stateErrorCovMtx_ - K * measMtx_ * stateErrorCovMtx_;

        // ShowFilterStatus();
        return x_hat_;
    }

    void ShowFilterStatus()
    {
        std::cout << "x\n";
        std::for_each(x_hat_.begin(), x_hat_.end(), [](const auto &val) { std::cout << val << " "; });
        std::cout << "\nP\n";
        std::cout << stateErrorCovMtx_ << std::endl;
    }

  public:
    Eigen::MatrixXd stateTransMtx_;    // n*n
    Eigen::MatrixXd measMtx_;          // m*n
    Eigen::MatrixXd measNoiseMtx_;     // m*m
    Eigen::MatrixXd processNoiseMtx_;  // n*n
    Eigen::MatrixXd stateErrorCovMtx_; // n*n

    Eigen::VectorXd x_hat_;

  private:
    const int32_t stateSize_; // n
    const int32_t measSize_;  // m
};