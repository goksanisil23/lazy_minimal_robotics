#include <functional>
#include <memory>

#include "landmarksim2d_msgs/msg/control_input_meas_msg.hpp"
#include "landmarksim2d_msgs/msg/range_bearing_obs_msg.hpp"
#include "rclcpp/rclcpp.hpp"

#include "ParticleFilter.h"

const std::string MAP_PATH = "/tmp/landmark_map_2d.txt";

class ParticleFilterNode : public rclcpp::Node
{
  public:
    ParticleFilterNode() : Node("particle_filter")
    {
        ctrlInMeasSub_ = this->create_subscription<landmarksim2d_msgs::msg::ControlInputMeasMsg>(
            "ctrl_in_meas",
            rclcpp::SensorDataQoS(),
            std::bind(&ParticleFilterNode::ControlInputMeasCallback, this, std::placeholders::_1));

        landmarkRangeBearingObsSub_ = this->create_subscription<landmarksim2d_msgs::msg::RangeBearingObsMsg>(
            "landmark_obs",
            rclcpp::SensorDataQoS(),
            std::bind(&ParticleFilterNode::LandmarkRangeBearingObsCallback, this, std::placeholders::_1));

        particleFilter_ = std::make_unique<ParticleFilter>(MAP_PATH);
    }

    void ControlInputMeasCallback(const landmarksim2d_msgs::msg::ControlInputMeasMsg &ctrlInMeasMsg)
    {
        std::cout << "ang vel: " << ctrlInMeasMsg.angular_vel << std::endl;
    }

    void LandmarkRangeBearingObsCallback(const landmarksim2d_msgs::msg::RangeBearingObsMsg &rangeBearingObsMsg)
    {
        std::cout << "range: " << rangeBearingObsMsg.ranges.at(0) << std::endl;
    }

  private:
    rclcpp::Subscription<landmarksim2d_msgs::msg::ControlInputMeasMsg>::SharedPtr ctrlInMeasSub_;
    rclcpp::Subscription<landmarksim2d_msgs::msg::RangeBearingObsMsg>::SharedPtr  landmarkRangeBearingObsSub_;

    std::unique_ptr<ParticleFilter> particleFilter_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ParticleFilterNode>());
    rclcpp::shutdown();

    return 0;
}
