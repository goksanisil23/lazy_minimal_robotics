#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/iteration_callback.h>
#include <ceres/loss_function.h>
#include <ceres/types.h>

class KalmanTuner : public std::enable_shared_from_this<KalmanTuner>
{
  public:
    KalmanTuner(const Eigen::VectorXd &x0);

    Eigen::MatrixXd Predict();

    Eigen::MatrixXd Correct(const Eigen::VectorXd &meas);

    void Tune(const std::vector<cv::Rect> &gtBboxes);

    // returns c_x,c_y,v_x,v_y,w,h
    static Eigen::VectorXd GetStateFromBbox(const cv::Rect2d &bbox);

    // returns center_x, center_y, width, height
    inline Eigen::VectorXd GetMeasFromBbox(const cv::Rect2d &bbox) const;

    // system dimensions
    static const int32_t stateSize_{6}; // c_x,c_y,v_x,v_y,w,h (center of bbox, velocity of bbox, height, width)
    static const int32_t measSize_{4};  // c_x, c_y, w, h
    static const int32_t ctrlSize{0};   // we dont have any control parameter in this case

    // Estimated states
    Eigen::VectorXd x_hat_; // c_x,c_y,v_x,v_y,w,h (center of bbox, velocity of bbox, height, width)
};