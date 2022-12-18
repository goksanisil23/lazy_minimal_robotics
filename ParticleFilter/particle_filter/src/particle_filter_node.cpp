#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

#include "LandmarkSim2dLib/Sim.h"
#include "landmarksim2d_msgs/msg/control_input_meas_msg.hpp"
#include "landmarksim2d_msgs/msg/range_bearing_obs_msg.hpp"
#include "landmarksim2d_msgs/srv/map.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "ParticleFilter.h"
#include "TimeUtil.h"

constexpr int16_t NUM_PARTICLES = 5000;

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

        particleVizPub_ = this->create_publisher<visualization_msgs::msg::Marker>("particles", rclcpp::SensorDataQoS());

        bestParticleVizPub_ =
            this->create_publisher<visualization_msgs::msg::Marker>("best_particle", rclcpp::SensorDataQoS());

        mapClient_ = this->create_client<landmarksim2d_msgs::srv::Map>("map_service");
    }

    void Start()
    {
        const std::string mapPath{GetMapInfo(mapClient_)};

        particleFilter_ =
            std::make_unique<ParticleFilter>(mapPath, NUM_PARTICLES, landmarkSim2D::Robot::GetSensorRange());
    }

    std::string GetMapInfo(std::shared_ptr<rclcpp::Client<landmarksim2d_msgs::srv::Map>> mapClient)
    {
        while (!mapClient->wait_for_service(std::chrono::seconds(2)))
        {
            RCLCPP_INFO(this->get_logger(), "Map server not available yet, waiting ...");
        }
        RCLCPP_INFO(this->get_logger(), "Discovered map service");
        auto mapRequest         = std::make_shared<landmarksim2d_msgs::srv::Map::Request>();
        mapRequest->map_request = true;
        auto resultFuture       = mapClient->async_send_request(mapRequest);
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), resultFuture) !=
            rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_ERROR(this->get_logger(), "Map service call failed");
            exit(1);
        }
        auto mapResponse = resultFuture.get();
        std::cout << "received map: " << mapResponse->map_path << std::endl;
        return std::string(mapResponse->map_path);
    }

    void ControlInputMeasCallback(const landmarksim2d_msgs::msg::ControlInputMeasMsg &ctrlInMeasMsg)
    {
        static auto prevTime = this->get_clock()->now();
        auto        currTime = this->get_clock()->now();
        auto        dt       = (currTime - prevTime).seconds();
        prevTime             = currTime;

        mutexParticle_.lock();
        {
            auto t1 = time_util::chronoNow();
            particleFilter_->PredictAndExplore(
                landmarkSim2D::ControlInput{ctrlInMeasMsg.linear_vel, ctrlInMeasMsg.angular_vel}, dt);
            auto t2 = time_util::chronoNow();
            PublishParticleViz(particleFilter_->particles_);
            auto t3 = time_util::chronoNow();

            time_util::showTimeDuration(t2, t1, "predict: ");
            time_util::showTimeDuration(t3, t2, "particle vis: ");
        }
        mutexParticle_.unlock();
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
        mutexParticle_.lock();
        {
            auto t1 = time_util::chronoNow();
            particleFilter_->UpdateWeightsWithObservations(landmarkObservations);
            // particleFilter_->UpdateWeightsWithObservations2(landmarkObservations);
            auto t2 = time_util::chronoNow();
            ShowBestParticle(particleFilter_->bestParticles_, particleFilter_->particles_);
            auto t3 = time_util::chronoNow();
            time_util::showTimeDuration(t2, t1, "update weight: ");
            time_util::showTimeDuration(t3, t2, "best particle: ");
            particleFilter_->CheckFilterReset();
        }
        mutexParticle_.unlock();
    }

    void PublishParticleViz(const std::vector<ParticleFilter::Particle> &particles)
    {
        auto particleMarkers = visualization_msgs::msg::Marker();
        // Generic marker properties
        particleMarkers.header.frame_id = "map";
        particleMarkers.header.stamp    = this->now();
        particleMarkers.ns              = "particles";
        particleMarkers.id              = 1;
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

    void ShowBestParticle(const std::multimap<float, size_t, std::greater<float>> &bestParticles,
                          const std::vector<ParticleFilter::Particle>             &particles)
    {
        // A) Find best particle
        // float  highestWeight{0.0};
        // size_t bestParticleIdx;
        // for (size_t partIdx = 0; partIdx < particles.size(); partIdx++)
        // {
        //     const auto particleWeight{particles.at(partIdx).weight};
        //     if (particleWeight > highestWeight)
        //     {
        //         highestWeight   = particleWeight;
        //         bestParticleIdx = partIdx;
        //     }
        // }
        // ParticleFilter::Particle bestParticle(particles.at(bestParticleIdx));

        // B) Average best particles
        ParticleFilter::Particle bestParticle;
        bestParticle.pose.yawRad = 0.0;
        bestParticle.pose.posX   = 0.0;
        bestParticle.pose.posY   = 0.0;
        for (const auto &bestParticlesItr : bestParticles)
        {
            bestParticle.pose.posX += particles.at(bestParticlesItr.second).pose.posX;
            bestParticle.pose.posY += particles.at(bestParticlesItr.second).pose.posY;
            bestParticle.pose.yawRad += particles.at(bestParticlesItr.second).pose.yawRad;
            // std::cout << bestParticlesItr.first << std::endl;
            // std::cout << particles.at(bestParticlesItr.second).beliefError2 << std::endl;
        }
        // std::cout << "----------------" << std::endl;
        bestParticle.pose.posX /= static_cast<float>(bestParticles.size());
        bestParticle.pose.posY /= static_cast<float>(bestParticles.size());
        bestParticle.pose.yawRad /= static_cast<float>(bestParticles.size());

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
        pointParticle.position.x = bestParticle.pose.posX;
        pointParticle.position.y = bestParticle.pose.posY;
        pointParticle.position.z = 0.0;

        bestParticleMarker.pose = pointParticle;

        bestParticleVizPub_->publish(bestParticleMarker);
    }

  private:
    rclcpp::Subscription<landmarksim2d_msgs::msg::ControlInputMeasMsg>::SharedPtr ctrlInMeasSub_;
    rclcpp::Subscription<landmarksim2d_msgs::msg::RangeBearingObsMsg>::SharedPtr  landmarkRangeBearingObsSub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr                 particleVizPub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr                 bestParticleVizPub_;
    rclcpp::Client<landmarksim2d_msgs::srv::Map>::SharedPtr                       mapClient_;

    std::unique_ptr<ParticleFilter> particleFilter_;

    std::mutex mutexParticle_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto pfNode = std::make_shared<ParticleFilterNode>();
    pfNode->Start();
    rclcpp::spin(pfNode);
    // rclcpp::spin(std::make_shared<ParticleFilterNode>());
    rclcpp::shutdown();

    return 0;
}
