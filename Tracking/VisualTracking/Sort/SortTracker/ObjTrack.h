#pragma once

#include <memory>

#include "KalmanFilter.hpp"

// Track associated to a unique object. As long as the object is being tracked, its motion is
// predicted by the motion model in Kalman filter and upon succesfull association with a detection,
// Kalman filter is updated with that detection.
class ObjTrack
{
  public:
    ObjTrack(const size_t &id, const Eigen::VectorXd &bbox);

    // Convert [c_x, c_y, w, h] to [c_x,c_y,v_x,v_y,w,h]
    Eigen::VectorXd ConvertMeasToState(const Eigen::VectorXd &bbox) const;
    Eigen::VectorXd GetPredBbox() const;

    void InitKfParameters(KalmanFilter &kf);
    void Predict(const double &dt);
    void Correct(const Eigen::VectorXd &measurement);

    void Draw() const;

  public:
    // state = [c_x,c_y,v_x,v_y,w,h] --> (center of bbox, velocity of bbox, height, width)
    // measurement = [c_x, c_y, w, h]
    std::unique_ptr<KalmanFilter> kf_;

  private:
    size_t id_;
};