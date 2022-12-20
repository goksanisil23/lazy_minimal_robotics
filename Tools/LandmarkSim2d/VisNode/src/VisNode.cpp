#include "VisNode.h"

#include <functional>

#include <tf2/transform_datatypes.h>

namespace landmarkSim2D
{

VisNode::VisNode() : Node("landmark_sim_2d")
{
    landmarkMapMarkers_ = visualization_msgs::msg::MarkerArray();

    // Create Sim object
    landmarkSim_ = std::make_unique<Sim>();

    robot_motion_timer_      = create_wall_timer(std::chrono::milliseconds(ROBOT_MOTION_PERIOD_MS),
                                            std::bind(&VisNode::MotionTimerCallback, this));
    robot_observation_timer_ = create_wall_timer(std::chrono::milliseconds(ROBOT_SENSING_PERIOD_MS),
                                                 std::bind(&VisNode::ObservationTimerCallback, this));

    truePosePublisher_ = this->create_publisher<nav_msgs::msg::Odometry>("gt_odom", rclcpp::SensorDataQoS());
    estPosePublisher_  = this->create_publisher<nav_msgs::msg::Odometry>("est_odom", rclcpp::SensorDataQoS());
    drPosePublisher_   = this->create_publisher<nav_msgs::msg::Odometry>("dr_odom", rclcpp::SensorDataQoS());
    landmarkMarkerPublisher_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>("landmark_markers", rclcpp::SensorDataQoS());
    observationMarkerPublisher_ =
        this->create_publisher<visualization_msgs::msg::Marker>("observation_markers", rclcpp::SensorDataQoS());
    sensorFovMarkerPublisher_ =
        this->create_publisher<visualization_msgs::msg::Marker>("sensor_fov_marker", rclcpp::SensorDataQoS());
    landmarkObsPublisher_ =
        this->create_publisher<landmarksim2d_msgs::msg::RangeBearingObsMsg>("landmark_obs", rclcpp::SensorDataQoS());
    ctrlInMeasPublisher_ =
        this->create_publisher<landmarksim2d_msgs::msg::ControlInputMeasMsg>("ctrl_in_meas", rclcpp::SensorDataQoS());
    mapServer_ = this->create_service<landmarksim2d_msgs::srv::Map>(
        "map_service", std::bind(&VisNode::MapServerHandler, this, std::placeholders::_1, std::placeholders::_2));
    trueTfPublisher_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
}

void VisNode::MapServerHandler(const std::shared_ptr<landmarksim2d_msgs::srv::Map::Request>  mapRequest,
                               const std::shared_ptr<landmarksim2d_msgs::srv::Map::Response> mapResponse)
{
    RCLCPP_INFO(this->get_logger(), "Received map request");
    if (mapRequest->map_request)
    {
        mapResponse->map_available = true;
        mapResponse->map_path      = landmarkSim2D::MAP_PATH;
    }

    PublishLandmarkMarkers(landmarkSim_->map->landmarks);
}

void VisNode::PublishLandmarkMarkers(const std::vector<Map::Landmark> &mapLandmarks)
{
    // Wait until rviz is connected before publishing landmark viz
    bool rvizConnected = false;
    while (!rvizConnected)
    {
        std::vector<rclcpp::TopicEndpointInfo> subscriptionInfo =
            this->get_subscriptions_info_by_topic("landmark_markers");
        for (auto subInfoNode : subscriptionInfo)
        {
            if (subInfoNode.node_name() == "rviz2")
            {
                RCLCPP_INFO(this->get_logger(), "Rviz is connected!");
                rvizConnected = true;
                break;
            }
        }
        if (!rvizConnected)
        {
            RCLCPP_INFO(this->get_logger(), "Waiting until Rviz is connected ...");
            std::this_thread::sleep_for(1s);
        }
    }

    // Generate rviz marker objects
    auto landmarkMarker = visualization_msgs::msg::Marker();
    // Generic marker properties
    landmarkMarker.header.frame_id = "map";
    landmarkMarker.header.stamp    = this->now();
    landmarkMarker.ns              = "landmarks";
    landmarkMarker.type            = visualization_msgs::msg::Marker::CYLINDER;
    landmarkMarker.action          = visualization_msgs::msg::Marker::ADD;
    landmarkMarker.lifetime        = rclcpp::Duration::from_nanoseconds(0); // forever
    landmarkMarker.scale.x         = 0.1;
    landmarkMarker.scale.y         = 0.1;
    landmarkMarker.scale.z         = 0.2;
    landmarkMarker.color.a         = 1.0;
    landmarkMarker.color.r         = 1.0;
    landmarkMarker.color.g         = 0.0;
    landmarkMarker.color.b         = 0.0;

    int16_t lmIdx = 0;
    for (auto &landmark : mapLandmarks)
    {
        landmarkMarker.id                 = lmIdx;
        landmarkMarker.pose.position.x    = landmark.posX;
        landmarkMarker.pose.position.y    = landmark.posY;
        landmarkMarker.pose.position.z    = 0.0;
        landmarkMarker.pose.orientation.x = 0.;
        landmarkMarker.pose.orientation.y = 0.;
        landmarkMarker.pose.orientation.z = 0.;
        landmarkMarker.pose.orientation.w = 1.;

        landmarkMapMarkers_.markers.push_back(landmarkMarker);
        lmIdx++;
    }
    landmarkMarkerPublisher_->publish(landmarkMapMarkers_);
}

//  Robot kinematics action and odometry visualization
void VisNode::MotionTimerCallback()
{
    static auto prevTime = this->get_clock()->now();
    auto        currTime = this->get_clock()->now();
    auto        dt       = (currTime - prevTime).seconds();
    landmarkSim_->Step(dt);
    prevTime = currTime;

    PublishTruePose(landmarkSim_->robot->truePose_);
    PublishDrPose(landmarkSim_->robot->drPose_);
    PublishCtrlInMeas(landmarkSim_->robot->measuredControlInput_);
}

// Robot sensing and visualization
void VisNode::ObservationTimerCallback()
{
    landmarkSim_->robot->Sense();
    ShowObservations(landmarkSim_->robot->observedLandmarks_, landmarkSim_->robot->truePose_);
    ShowRobotFov(landmarkSim_->robot->truePose_, landmarkSim_->robot->GetSensorRange());
    PublishObservations(landmarkSim_->robot->observedLandmarks_);
}

void VisNode::PublishTruePose(const landmarkSim2D::Pose2D &robotPose)
{
    // Generate odometry msg from robot pose
    nav_msgs::msg::Odometry gtOdom;
    gtOdom.header.frame_id      = "map";
    gtOdom.child_frame_id       = "robot";
    gtOdom.header.stamp         = this->get_clock()->now();
    gtOdom.pose.pose.position.x = robotPose.posX;
    gtOdom.pose.pose.position.y = robotPose.posY;
    gtOdom.pose.pose.position.z = 0.0;
    tf2::Quaternion quat;
    quat.setRPY(0., 0., robotPose.yawRad);
    quat.normalize();
    gtOdom.pose.pose.orientation = tf2::toMsg(quat);

    truePosePublisher_->publish(gtOdom);

    // send tf
    geometry_msgs::msg::TransformStamped tfTrue;
    tfTrue.header                  = gtOdom.header;
    tfTrue.child_frame_id          = gtOdom.child_frame_id;
    tfTrue.transform.translation.x = robotPose.posX;
    tfTrue.transform.translation.y = robotPose.posY;
    tfTrue.transform.translation.z = 0;
    tf2::convert(quat, tfTrue.transform.rotation);
    trueTfPublisher_->sendTransform(tfTrue);
}

void VisNode::PublishDrPose(const landmarkSim2D::Pose2D &robotPose)
{
    // Generate odometry msg from robot pose
    nav_msgs::msg::Odometry drOdom;
    drOdom.header.frame_id      = "map";
    drOdom.child_frame_id       = "robot";
    drOdom.header.stamp         = this->get_clock()->now();
    drOdom.pose.pose.position.x = robotPose.posX;
    drOdom.pose.pose.position.y = robotPose.posY;
    drOdom.pose.pose.position.z = 0.0;
    tf2::Quaternion quat;
    quat.setRPY(0., 0., robotPose.yawRad);
    quat.normalize();
    drOdom.pose.pose.orientation = tf2::toMsg(quat);

    drPosePublisher_->publish(drOdom);
}

void VisNode::ShowObservations(const std::vector<RangeBearingObs> &landmarkObservations, const Pose2D &robotPose)
{
    auto obsMarkerLines = visualization_msgs::msg::Marker();
    // Generic marker properties
    obsMarkerLines.header.frame_id = "map";
    obsMarkerLines.header.stamp    = this->now();
    obsMarkerLines.ns              = "landmark_obs";
    obsMarkerLines.type            = visualization_msgs::msg::Marker::LINE_LIST;
    obsMarkerLines.action          = visualization_msgs::msg::Marker::ADD;
    // obsMarkerLines.lifetime        = rclcpp::Duration::from_nanoseconds(0); // forever
    obsMarkerLines.scale.x = 0.02;
    obsMarkerLines.scale.y = 0.1;
    obsMarkerLines.scale.z = 0.1;
    obsMarkerLines.color.a = 1.0;
    obsMarkerLines.color.r = 1.0;
    obsMarkerLines.color.g = 1.0;
    obsMarkerLines.color.b = 0.2;

    // Generate lines
    for (const auto &lmObs : landmarkObservations)
    {
        geometry_msgs::msg::Point p0, p1;
        p0.x = robotPose.posX;
        p0.y = robotPose.posY;
        p0.z = 0.;
        p1.x = p0.x + lmObs.range * cos(lmObs.angleRad + robotPose.yawRad);
        p1.y = p0.y + lmObs.range * sin(lmObs.angleRad + robotPose.yawRad);
        p1.z = 0.;

        obsMarkerLines.points.push_back(p0);
        obsMarkerLines.points.push_back(p1);
    }

    observationMarkerPublisher_->publish(obsMarkerLines);
}

void VisNode::PublishObservations(const std::vector<RangeBearingObs> &landmarkObservations)
{
    landmarksim2d_msgs::msg::RangeBearingObsMsg lmObsMsg;
    lmObsMsg.header.stamp    = this->now();
    lmObsMsg.header.frame_id = "robot";
    for (const auto &lmObs : landmarkObservations)
    {
        lmObsMsg.ids.push_back(lmObs.id);
        lmObsMsg.ranges.push_back(lmObs.range);
        lmObsMsg.angles.push_back(lmObs.angleRad);
    }

    landmarkObsPublisher_->publish(lmObsMsg);
}

void VisNode::PublishCtrlInMeas(const landmarkSim2D::ControlInput &controlInputMeas)
{
    landmarksim2d_msgs::msg::ControlInputMeasMsg ctrlInMeasMsg;

    ctrlInMeasMsg.header.stamp    = this->now();
    ctrlInMeasMsg.header.frame_id = "robot";
    ctrlInMeasMsg.linear_vel      = controlInputMeas.linearVel;
    ctrlInMeasMsg.angular_vel     = controlInputMeas.angularVel;

    ctrlInMeasPublisher_->publish(ctrlInMeasMsg);
}

void VisNode::ShowRobotFov(const Pose2D &robotPose, const float &sensorRange)
{
    static auto sensorFovMarker     = visualization_msgs::msg::Marker();
    sensorFovMarker.header.frame_id = "map";
    sensorFovMarker.header.stamp    = this->now();
    sensorFovMarker.ns              = "sensor_fov";
    sensorFovMarker.type            = visualization_msgs::msg::Marker::CYLINDER;
    sensorFovMarker.pose.position.x = robotPose.posX;
    sensorFovMarker.pose.position.y = robotPose.posY;
    sensorFovMarker.scale.x         = sensorRange * 2.0f;
    sensorFovMarker.scale.y         = sensorRange * 2.0f;
    sensorFovMarker.scale.z         = 0.01;
    sensorFovMarker.color.a         = 0.5;
    sensorFovMarker.color.r         = 0.5;
    sensorFovMarker.color.g         = 0.5;
    sensorFovMarker.color.b         = 0.5;

    sensorFovMarkerPublisher_->publish(sensorFovMarker);
}

} // namespace landmarkSim2D