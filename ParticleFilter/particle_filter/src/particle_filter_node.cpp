#include <functional>
#include <memory>

#include "landmarkSim2dLib/Sim.h"
#include "landmarksim2d_msgs/msg/control_input_meas_msg.hpp"
#include "landmarksim2d_msgs/msg/range_bearing_obs_msg.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "ParticleFilter.h"

const std::string MAP_PATH      = "/tmp/landmark_map_2d.txt";
constexpr int16_t NUM_PARTICLES = 1000;

class ParticleFilterNode : public rclcpp::Node
{
  public:
    ParticleFilterNode() : Node("particle_filter")
    {
        particleFilter_ =
            std::make_unique<ParticleFilter>(MAP_PATH, NUM_PARTICLES, landmarkSim2D::Robot::GetSensorRange());

        ctrlInMeasSub_ = this->create_subscription<landmarksim2d_msgs::msg::ControlInputMeasMsg>(
            "ctrl_in_meas",
            rclcpp::SensorDataQoS(),
            std::bind(&ParticleFilterNode::ControlInputMeasCallback, this, std::placeholders::_1));

        landmarkRangeBearingObsSub_ = this->create_subscription<landmarksim2d_msgs::msg::RangeBearingObsMsg>(
            "landmark_obs",
            rclcpp::SensorDataQoS(),
            std::bind(&ParticleFilterNode::LandmarkRangeBearingObsCallback, this, std::placeholders::_1));

        particleVizPub_ = this->create_publisher<visualization_msgs::msg::Marker>("particles", rclcpp::SensorDataQoS());

        bestParticleVizPub_ =
            this->create_publisher<visualization_msgs::msg::Marker>("best_particle", rclcpp::SensorDataQoS());
    }

    void ControlInputMeasCallback(const landmarksim2d_msgs::msg::ControlInputMeasMsg &ctrlInMeasMsg)
    {
        static auto prevTime = this->get_clock()->now();
        auto        currTime = this->get_clock()->now();
        auto        dt       = (currTime - prevTime).seconds();
        prevTime             = currTime;

        particleFilter_->PredictAndExplore(
            landmarkSim2D::ControlInput{ctrlInMeasMsg.linear_vel, ctrlInMeasMsg.angular_vel}, dt);
        PublishParticleViz(particleFilter_->particles_);
    }

    void LandmarkRangeBearingObsCallback(const landmarksim2d_msgs::msg::RangeBearingObsMsg &rangeBearingObsMsg)
    {
        std::vector<landmarkSim2D::RangeBearingObs> landmarkObservations;
        for (size_t lmObsIdx = 0; lmObsIdx < rangeBearingObsMsg.angles.size(); lmObsIdx++)
        {
            landmarkObservations.push_back(landmarkSim2D::RangeBearingObs(rangeBearingObsMsg.ids.at(lmObsIdx),
                                                                          rangeBearingObsMsg.ranges.at(lmObsIdx),
                                                                          rangeBearingObsMsg.angles.at(lmObsIdx)));
        }
        particleFilter_->UpdateWeightsWithObservations(landmarkObservations);
        ShowBestParticle(particleFilter_->particles_);
    }

    void PublishParticleViz(const std::vector<ParticleFilter::Particle> &particles)
    {

        // Generate rviz marker objects
        auto particleMarkers = visualization_msgs::msg::Marker();
        // Generic marker properties
        particleMarkers.header.frame_id = "map";
        particleMarkers.header.stamp    = this->now();
        particleMarkers.ns              = "particles";
        particleMarkers.type            = visualization_msgs::msg::Marker::SPHERE_LIST;
        particleMarkers.action          = visualization_msgs::msg::Marker::ADD;
        // particleMarkers.lifetime        = rclcpp::Duration::from_nanoseconds(0); // forever
        particleMarkers.scale.x = 0.06;
        particleMarkers.scale.y = 0.06;
        particleMarkers.scale.z = 0.06;
        particleMarkers.color.a = 1.0;
        particleMarkers.color.r = 1.0;
        particleMarkers.color.g = 1.0;
        particleMarkers.color.b = 1.0;

        for (const auto &particle : particles)
        {
            geometry_msgs::msg::Point pointParticle;
            pointParticle.x = particle.pose.posX;
            pointParticle.y = particle.pose.posY;
            pointParticle.z = 0.0;

            particleMarkers.points.push_back(pointParticle);
        }
        particleVizPub_->publish(particleMarkers);
    }

    void ShowBestParticle(const std::vector<ParticleFilter::Particle> &particles)
    {
        // Find best particle
        float  highestWeight{0.0};
        size_t bestParticleIdx;
        for (size_t partIdx = 0; partIdx < particles.size(); partIdx++)
        {
            const auto particleWeight{particles.at(partIdx).weight};
            if (particleWeight > highestWeight)
            {
                highestWeight   = particleWeight;
                bestParticleIdx = partIdx;
            }
        }

        // Generate rviz marker objects
        auto bestParticleMarker = visualization_msgs::msg::Marker();
        // Generic marker properties
        bestParticleMarker.header.frame_id = "map";
        bestParticleMarker.header.stamp    = this->now();
        bestParticleMarker.ns              = "best_particle";
        bestParticleMarker.type            = visualization_msgs::msg::Marker::SPHERE;
        bestParticleMarker.action          = visualization_msgs::msg::Marker::ADD;
        // bestParticleMarker.lifetime        = rclcpp::Duration::from_nanoseconds(0); // forever
        bestParticleMarker.scale.x = 0.5;
        bestParticleMarker.scale.y = 0.5;
        bestParticleMarker.scale.z = 0.5;
        bestParticleMarker.color.a = 1.0;
        bestParticleMarker.color.r = 0.0;
        bestParticleMarker.color.g = 0.3;
        bestParticleMarker.color.b = 1.0;

        geometry_msgs::msg::Pose pointParticle;
        pointParticle.position.x = particles.at(bestParticleIdx).pose.posX;
        pointParticle.position.y = particles.at(bestParticleIdx).pose.posY;
        pointParticle.position.z = 0.0;

        bestParticleMarker.pose = pointParticle;

        bestParticleVizPub_->publish(bestParticleMarker);
    }

  private:
    rclcpp::Subscription<landmarksim2d_msgs::msg::ControlInputMeasMsg>::SharedPtr ctrlInMeasSub_;
    rclcpp::Subscription<landmarksim2d_msgs::msg::RangeBearingObsMsg>::SharedPtr  landmarkRangeBearingObsSub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr                 particleVizPub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr                 bestParticleVizPub_;

    std::unique_ptr<ParticleFilter> particleFilter_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ParticleFilterNode>());
    rclcpp::shutdown();

    return 0;
}
