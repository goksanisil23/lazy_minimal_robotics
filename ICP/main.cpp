#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <mutex>

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

// #include <pcl/visualization/cloud_viewer.h>

#include "open3d/Open3D.h"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

#define EXPECT_TRUE(pred) if (!(pred)) { throw std::runtime_error(#pred); }

std::mutex cloud_mtx;
std::condition_variable cloud_sig;

int main() {
  try {

    std::string host("localhost");
    uint16_t port(2000u);

    auto client = cc::Client(host, port);
    client.SetTimeout(40s);

    std::cout << "Client API version : " << client.GetClientVersion() << '\n';
    std::cout << "Server API version : " << client.GetServerVersion() << '\n';

    cc::World world = client.GetWorld();
    auto settings = world.GetSettings();
    settings.fixed_delta_seconds = 0.1;
    // settings.synchronous_mode = true;
    world.ApplySettings(settings, 3s);

    // Get a random vehicle blueprint.
    auto blueprint_library = world.GetBlueprintLibrary();
    auto vehicle_bp = blueprint_library->Find("vehicle.volkswagen.t2");
    // // auto blueprint = RandomChoice(*vehicles, rng);

    // // Find a valid spawn point.
    auto map = world.GetMap();
    auto transform = map->GetRecommendedSpawnPoints()[0];

    // // Spawn the vehicle.
    auto actor = world.SpawnActor(*vehicle_bp, transform);
    std::cout << "Spawned " << actor->GetDisplayId() << '\n';
    auto vehicle = boost::static_pointer_cast<cc::Vehicle>(actor);

    // Find a lidar blueprint.
    auto lidar_bp = (*(blueprint_library->Filter("sensor.lidar.ray_cast")))[0];
    // Spawn a lidar attached to the vehicle.
    auto lidar_transform = cg::Transform{
        cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
        cg::Rotation{0.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.

    // auto &attribute = lidar_bp->GetAttribute("upper_fov");
    lidar_bp.SetAttribute("upper_fov", "22.5");

    lidar_bp.SetAttribute("dropoff_general_rate", "0.0");
    lidar_bp.SetAttribute("dropoff_intensity_limit", "1.0");
    lidar_bp.SetAttribute("dropoff_zero_intensity", "0.0");
    lidar_bp.SetAttribute("upper_fov", "22.5");
    lidar_bp.SetAttribute("lower_fov", "-22.5");
    lidar_bp.SetAttribute("channels", "64");
    lidar_bp.SetAttribute("range", "150.0");
    lidar_bp.SetAttribute("points_per_second", "327680");
    lidar_bp.SetAttribute("rotation_frequency", "10");
    // lidar_bp.SetAttribute("sensor_tick", "0.1");
    auto lidar_actor = world.SpawnActor(lidar_bp, lidar_transform, actor.get());   
    auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);

    // // Apply control to vehicle.
    cc::Vehicle::Control control;
    control.throttle = 1.0f;
    vehicle->ApplyControl(control);

    // // Move spectator so we can see the vehicle from the simulator window.
    auto spectator = world.GetSpectator();
    transform.location += 32.0f * transform.GetForwardVector();
    transform.location.z += 2.0f;
    transform.rotation.yaw += 180.0f;
    transform.rotation.pitch = -15.0f;
    spectator->SetTransform(transform);

    // setup visualization
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("Lidar data", 960, 540, 480, 270);
    vis.GetRenderOption().background_color_ = {0.05, 0.05, 0.05};
    vis.GetRenderOption().point_size_ = 1;
    vis.GetRenderOption().show_coordinate_frame_ = true;

    std::vector<Eigen::Vector3d> o3d_points;

    // Register a callback to listen to lidar data.
    lidar->Listen([&o3d_points](auto lidar_data) {
      auto pointcloud = boost::static_pointer_cast<csd::Array<csd::LidarDetection>>(lidar_data);
      cloud_mtx.lock();
        o3d_points.clear();
        for(auto& pt : *pointcloud) {
          o3d_points.push_back({pt.point.x, pt.point.y, pt.point.z});
        }
      cloud_mtx.unlock();
      cloud_sig.notify_all();
    });

    std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud;
    int32_t pt_ctr = 0;
    while(true) {
      // cloud_mtx.lock();
      if(pt_ctr == 100) {
        std::cout << "Created point cloud" << std::endl;
        cloud_mtx.lock();
        o3d_cloud = std::make_shared<open3d::geometry::PointCloud>(o3d_points);
        vis.AddGeometry(o3d_cloud);
        cloud_mtx.unlock();
        cloud_sig.notify_all();          
      }
      else if (pt_ctr > 100) {
        std::cout << o3d_points.size() << std::endl;
        cloud_mtx.lock();
        o3d_cloud->points_ = o3d_points;
        vis.UpdateGeometry(o3d_cloud);
        cloud_mtx.unlock();
        cloud_sig.notify_all();      
        vis.PollEvents();
        vis.UpdateRender();
      }

      pt_ctr++;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // std::this_thread::sleep_for(10s);

    // Remove actors from the simulation.
    lidar->Destroy();
    vehicle->Destroy(); 



/*
    // Find a camera blueprint.
    auto camera_bp = blueprint_library->Find("sensor.camera.semantic_segmentation");
    EXPECT_TRUE(camera_bp != nullptr);

    // Spawn a camera attached to the vehicle.
    auto camera_transform = cg::Transform{
        cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
        cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, actor.get());
    auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);

    // Register a callback to save images to disk.
    camera->Listen([](auto data) {
        auto image = boost::static_pointer_cast<csd::Image>(data);
        EXPECT_TRUE(image != nullptr);
        SaveSemSegImageToDisk(*image);
    });

    // std::this_thread::sleep_for(10s);

    // Remove actors from the simulation.
    camera->Destroy();
*/
    // vehicle->Destroy();
    // std::cout << "Actors destroyed." << std::endl;

  } catch (const cc::TimeoutException &e) {
    std::cout << '\n' << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cout << "\nException: " << e.what() << std::endl;
    return 2;
  }
}
