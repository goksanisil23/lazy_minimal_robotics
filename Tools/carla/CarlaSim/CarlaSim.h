#pragma once

#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Client.h>
#include <carla/client/Map.h>
#include <carla/client/Sensor.h>
#include <carla/client/TimeoutException.h>
#include <carla/client/World.h>
#include <carla/geom/Transform.h>
#include <carla/image/ImageIO.h>
#include <carla/image/ImageView.h>
#include <carla/sensor/data/Image.h>
#include <carla/sensor/data/LidarData.h>

#include "Eigen/Dense"
#include "open3d/Open3D.h"

#include "ColorPalette.hpp"
#include "ThreadSafeQueue.h"

namespace cc  = carla::client;
namespace cg  = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

using SemanticLidarData = csd::Array<csd::SemanticLidarDetection>;

constexpr double SIM_STEP_TIME = 0.1; // secs

class CarlaSim
{
  public:
    CarlaSim() = default;

    void                           Setup();
    boost::shared_ptr<cc::Vehicle> SpawnVehicle();
    boost::shared_ptr<cc::Sensor>  SpawnLidar();
    void                           MoveSpectator();
    uint64_t                       Step();
    void                           LidarCallback(boost::shared_ptr<carla::sensor::SensorData> lidarDataPtr);
    void                           GetLidarData(boost::shared_ptr<SemanticLidarData> &lidarDataPtr);
    void                           GetLidarPose(Eigen::Matrix4f &lidarPose);
    void                           Terminate();
    void                           SetupCloudViz();
    void                           UpdateCloudViz(boost::shared_ptr<SemanticLidarData> pointcloudPtr);
    static void CarlaToRoboticsTransform(const cg::Transform &carlaPose, Eigen::Matrix4f &roboticsPose);

  private:
    std::shared_ptr<cc::Client> client_;
    std::shared_ptr<cc::World>  world_;

    carla::rpc::EpisodeSettings originalSettings_;

    boost::shared_ptr<cc::Vehicle>          vehicle_;
    boost::shared_ptr<cc::BlueprintLibrary> blueprintLib_;
    boost::shared_ptr<cc::Sensor>           lidar_;
    boost::shared_ptr<cc::Actor>            spectator_;

    std::mutex              cloudMtx_;
    std::condition_variable cloudSig_;

    std::vector<Eigen::Vector3d>                  o3dPoints_;
    std::vector<Eigen::Vector3d>                  o3dColors_;
    std::shared_ptr<open3d::geometry::PointCloud> o3dCloud_;
    open3d::visualization::Visualizer             o3dCloudVis_;
    ColorPalette                                  semSegColors_;

    // Thread-safe queue for storing the captured sensor data
    TSQueue<boost::shared_ptr<SemanticLidarData>> lidarDataPtrQueue_;
};