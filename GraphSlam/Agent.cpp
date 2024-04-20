#include "Agent.h"

#include "SimConstants.h"

Agent::Agent(const raylib::Vector2 position_initial, const float heading_initial)
    : position_{position_initial}, heading_{heading_initial}
{
}

std::pair<float, float> Agent::readIMU(float linearVelocity, float angularVelocity)
{
    return {
        linearVelocity + linear_vel_noise_dist_(noise_generator_),  // Noisy linear velocity
        angularVelocity + angular_vel_noise_dist_(noise_generator_) // Noisy angular velocity
    };
}

void Agent::updateMovement(float linearVelocity, float angularVelocity)
{
    heading_ += angularVelocity * kDt;
    position_.x += linearVelocity * kDt * cos(heading_);
    position_.y += linearVelocity * kDt * sin(heading_);
}

void Agent::senseLandmarks(const std::vector<Landmark> &landmarks)
{
    landmark_measurements_.clear();

    for (const auto &landmark : landmarks)
    {
        float distance = position_.Distance(landmark.position_);
        float bearing  = atan2(landmark.position_.y - position_.y, landmark.position_.x - position_.x) - heading_;
        if (distance <= kSensorRange)
        {
            // Simulate measurement noise for distance and bearing
            distance += landmark_range_noise_dist_(noise_generator_);
            bearing += landmark_bearing_noise_dist_(noise_generator_);
            landmark_measurements_.push_back({distance, bearing, landmark.id_});
        }
    }
}
