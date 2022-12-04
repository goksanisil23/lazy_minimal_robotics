#pragma once

#include <chrono>
#include <memory>

#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "Sim.h"

using namespace std::chrono_literals;

namespace landmarkSim2D
{
class VisNode : public rclcpp::Node
{
  public:
    VisNode();

  private:
    void MotionTimerCallback();
    void ObservationTimerCallback();
    void PublishLandmarkMarkers(const std::vector<Map::Landmark> &mapLandmarks);
    void PublishGtOdom(const landmarkSim2D::Pose2D &robotPose);
    void ShowObservations(const std::vector<RangeBearingObs> &landmarkObservations, const Pose2D &robotPose);
    void ShowRobotFov(const Pose2D &robotPose, const float &sensorRange);

    std::unique_ptr<Sim> landmarkSim_;

    rclcpp::TimerBase::SharedPtr robot_motion_timer_;
    rclcpp::TimerBase::SharedPtr robot_observation_timer_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr landmarkMarkerPublisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr      observationMarkerPublisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr      sensorFovMarkerPublisher_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr gtOdomPublisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr deadreckonOdomPublisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr estOdomPublisher_;

    visualization_msgs::msg::MarkerArray landmarkMapMarkers_;
};

} // namespace landmarkSim2D