#include "Odometry.h"

#include "SimConstants.h"

// Updates delta movement in local-robot frame
void Odometry::update(const float &linear_vel_meas, const float &angular_vel_meas)
{
    constexpr float kAngularVelThresh{1e-6};
    if (fabs(angular_vel_meas) < kAngularVelThresh)
    {
        delta_position_.x = linear_vel_meas * kDt;
        delta_position_.y = 0;
        delta_heading_    = 0;
    }
    else
    {
        float R           = linear_vel_meas / angular_vel_meas; // turning radius
        delta_heading_    = angular_vel_meas * kDt;
        delta_position_.x = R * sin(delta_heading_);
        delta_position_.y = R * (1 - cos(delta_heading_));
    }
}