#pragma once

#include <chrono>
#include <memory>

#include "landmarksim2d_msgs/msg/control_input_meas_msg.hpp"
#include "landmarksim2d_msgs/msg/range_bearing_obs_msg.hpp"
#include "landmarksim2d_msgs/srv/map.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <LandmarkSim2dLib/Sim.h>

using namespace std::chrono_literals;

namespace landmarkSim2D
{
// constexpr uint32_t ROBOT_MOTION_PERIOD_MS  = 100;
// constexpr uint32_t ROBOT_SENSING_PERIOD_MS = 100;
constexpr uint32_t ROBOT_MOTION_PERIOD_MS  = 400;
constexpr uint32_t ROBOT_SENSING_PERIOD_MS = 400;

class VisNode : public rclcpp::Node
{
  public:
    VisNode();

  private:
    void MotionTimerCallback();
    void ObservationTimerCallback();
    void PublishLandmarkMarkers(const std::vector<Map::Landmark> &mapLandmarks);
    void PublishTruePose(const landmarkSim2D::Pose2D &robotPose);
    void PublishDrPose(const landmarkSim2D::Pose2D &robotPose);
    void PublishCtrlInMeas(const landmarkSim2D::ControlInput &controlInputMeas);
    void PublishObservations(const std::vector<RangeBearingObs> &landmarkObservations);
    void ShowObservations(const std::vector<RangeBearingObs> &landmarkObservations, const Pose2D &robotPose);
    void ShowRobotFov(const Pose2D &robotPose, const float &sensorRange);
    void MapServerHandler(const std::shared_ptr<landmarksim2d_msgs::srv::Map::Request>  mapRequest,
                          const std::shared_ptr<landmarksim2d_msgs::srv::Map::Response> mapResponse);

    std::unique_ptr<Sim> landmarkSim_;

    rclcpp::TimerBase::SharedPtr robot_motion_timer_;
    rclcpp::TimerBase::SharedPtr robot_observation_timer_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr landmarkMarkerPublisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr      observationMarkerPublisher_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr      sensorFovMarkerPublisher_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr truePosePublisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr drPosePublisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr estPosePublisher_;

    rclcpp::Publisher<landmarksim2d_msgs::msg::RangeBearingObsMsg>::SharedPtr  landmarkObsPublisher_;
    rclcpp::Publisher<landmarksim2d_msgs::msg::ControlInputMeasMsg>::SharedPtr ctrlInMeasPublisher_;

    rclcpp::Service<landmarksim2d_msgs::srv::Map>::SharedPtr mapServer_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> trueTfPublisher_;

    visualization_msgs::msg::MarkerArray landmarkMapMarkers_;
};

} // namespace landmarkSim2D