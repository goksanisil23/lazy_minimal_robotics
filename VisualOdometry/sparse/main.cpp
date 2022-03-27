#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <mutex>
#include <queue>

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

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv4/opencv2/opencv.hpp>

#include "ThreadSafeQueue.h"
#include "Sparse.hpp"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

constexpr int16_t IMAGE_WIDTH = 1024;
constexpr int16_t IMAGE_HEIGHT = 640;
// constexpr float CAMERA_POS_X = 2.0;
constexpr float CAMERA_POS_X = -5.5;
constexpr float CAMERA_POS_Y = 0.0;
constexpr float CAMERA_POS_Z = 2.8;
constexpr u_int32_t NUM_SIM_STEPS = 1000;
constexpr double SIM_STEP_TIME = 0.1; // secs

#define EXPECT_TRUE(pred) if (!(pred)) { throw std::runtime_error(#pred); }

// Converts Carla RGB Camera image to OpenCV image, by popping the oldest element in the queue
void CarlaToOpenCV(TSQueue<boost::shared_ptr<csd::Image>>& carla_image_queue, cv::Mat& cv_img)
{
  boost::shared_ptr<csd::Image> carla_img_ptr;
  carla_image_queue.dequeue(carla_img_ptr);

    csd::Color* image_data = carla_img_ptr->data();
    
    for(int ii = 0; ii < IMAGE_HEIGHT; ii++)
    {
      for(int jj = 0; jj < IMAGE_WIDTH; jj++)
      {
        csd::Color color = image_data[jj + ii*IMAGE_WIDTH];
        cv::Vec3b& cv_color = cv_img.at<cv::Vec3b>(ii,jj); 
        cv_color[0] = color.b;
        cv_color[1] = color.g;
        cv_color[2] = color.r;
      }
    }
}

Eigen::Matrix3f getCameraIntrinsic(const carla::client::BlueprintLibrary::value_type& camera_bp)
{
  Eigen::Matrix3f K = Eigen::Matrix3f::Identity(); // intrinsics matrix for pinhole
  int image_w = camera_bp.GetAttribute("image_size_x").As<int>();
  int image_h = camera_bp.GetAttribute("image_size_y").As<int>();
  float fov = camera_bp.GetAttribute("fov").As<float>();
  float focal = image_w / (2.0 * std::tan(fov * M_PI / 360.0));
  K(0,0) = focal;
  K(1,1) = focal;
  K(0,2) = image_w/2.0;
  K(1,2) = image_h/2.0;
  return K;
}


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
    auto traffic_manager = client.GetInstanceTM();
    traffic_manager.SetSynchronousMode(true);
    settings.fixed_delta_seconds = SIM_STEP_TIME;
    settings.synchronous_mode = true;
    settings.no_rendering_mode = true;
    world.ApplySettings(settings, 3s);

    // Get a random vehicle blueprint.
    auto blueprint_library = world.GetBlueprintLibrary();
    auto vehicle_bp = blueprint_library->Find("vehicle.volkswagen.t2");

    // // Find a valid spawn point.
    auto map = world.GetMap();
    auto transform = map->GetRecommendedSpawnPoints()[1];

    // // Spawn the vehicle.
    auto actor = world.SpawnActor(*vehicle_bp, transform);
    std::cout << "Spawned " << actor->GetDisplayId() << '\n';
    auto vehicle = boost::static_pointer_cast<cc::Vehicle>(actor);
    vehicle->SetAutopilot(true);
    vehicle->OpenDoor(cc::Vehicle::VehicleDoor::All);

    // Find a camera blueprint.
    auto camera_rgb_bp = (*(blueprint_library->Filter("sensor.camera.rgb")))[0];
    // Spawn a camera attached to the vehicle.
    auto camera_transform = cg::Transform{
        cg::Location{CAMERA_POS_X, CAMERA_POS_Y, CAMERA_POS_Z},   // x, y, z.
        cg::Rotation{0.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.

    camera_rgb_bp.SetAttribute("image_size_x", std::to_string(IMAGE_WIDTH));
    camera_rgb_bp.SetAttribute("image_size_y", std::to_string(IMAGE_HEIGHT));

    auto camera_actor = world.SpawnActor(camera_rgb_bp, camera_transform, actor.get());   
    auto camera = boost::static_pointer_cast<cc::Sensor>(camera_actor);

    // Thread-safe queue for storing the captured images
    TSQueue<boost::shared_ptr<csd::Image>> carla_image_queue;
    camera->Listen([&](auto data)
    {
      boost::shared_ptr<csd::Image> image_ptr = boost::static_pointer_cast<csd::Image>(data);
      carla_image_queue.enqueue(image_ptr);
    });

    // Create the VISO instance
    VISO::Sparse viso_sparse(getCameraIntrinsic(camera_rgb_bp));

    cv::Mat cv_img(IMAGE_HEIGHT,IMAGE_WIDTH, CV_8UC3, cv::Scalar(0,0,0));
    // Advance the simulation
    for(int ii = 0; ii < NUM_SIM_STEPS; ii++)
    {
      auto frame_id = world.Tick(1s); // timeout

      CarlaToOpenCV(carla_image_queue, cv_img);
      viso_sparse.step(cv_img);
      // cv::imshow("carla bgr",cv_img);
      cv::waitKey(10);

      std::this_thread::sleep_for(1ms);
      std::cout << "sim frame id: " << frame_id << std::endl;
      auto camera_pose = camera->GetTransform();
      std::cout << camera_pose.location.x << " " << camera_pose.location.y << " " << camera_pose.location.z << std::endl;  
    }

    // Remove actors from the simulation.
    camera->Destroy();
    vehicle->Destroy(); 


  } catch (const cc::TimeoutException &e) {
    std::cout << '\n' << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cout << "\nException: " << e.what() << std::endl;
    return 2;
  }
}
