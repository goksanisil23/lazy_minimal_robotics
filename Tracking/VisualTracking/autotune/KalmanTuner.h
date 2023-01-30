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
    KalmanTuner();

    void Tune(const std::vector<cv::Rect> &gtBboxes);

    // returns c_x,c_y,v_x,v_y,w,h
    static Eigen::VectorXd Get6DStateFromBbox(const cv::Rect2d &bbox);

    // returns center_x, center_y, width, height
    static Eigen::VectorXd Get4DMeasFromBbox(const cv::Rect2d &bbox);

    // Retrieves opencv style rectangle from state
    static cv::Rect GetCvRectFromState(const Eigen::Vector<double, 6> &state);

    // system dimensions
    static const int32_t stateSize_{6}; // c_x,c_y,v_x,v_y,w,h (center of bbox, velocity of bbox, height, width)
    static const int32_t measSize_{4};  // c_x, c_y, w, h
    static const int32_t ctrlSize{0};   // we dont have any control parameter in this case

    std::vector<double> filterNoiseParams;
};