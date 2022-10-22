#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

// CARLA
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

// EIGEN
#include <Eigen/Dense>
#include <Eigen/Geometry>

// OPENCV
#include <opencv4/opencv2/opencv.hpp>

#include "ThreadSafeQueue.h"

#include "StereoDepth.hpp"

namespace cc  = carla::client;
namespace cg  = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

constexpr int16_t   IMAGE_WIDTH        = 1024;
constexpr int16_t   IMAGE_HEIGHT       = 640;
constexpr float     CAMERA_LEFT_POS_X  = 0.0;
constexpr float     CAMERA_LEFT_POS_Y  = 0.0;
constexpr float     CAMERA_LEFT_POS_Z  = 0.0;
constexpr float     CAMERA_RIGHT_POS_X = 0.0;  // forward
constexpr float     CAMERA_RIGHT_POS_Y = -1.0; // left
constexpr float     CAMERA_RIGHT_POS_Z = 0.0;  // up
constexpr u_int32_t NUM_SIM_STEPS      = 1000;
constexpr double    SIM_STEP_TIME      = 0.1; // secs

// Rectified projection matrices
cv::Mat P_left  = (cv::Mat_<float>(3, 4) << 640.0, 0.0, 640.0, 2176.0, 0.0, 480.0, 480.0, 552.0, 0.0, 0.0, 1.0, 1.4);
cv::Mat P_right = (cv::Mat_<float>(3, 4) << 640.0, 0.0, 640.0, 2176.0, 0.0, 480.0, 480.0, 792.0, 0.0, 0.0, 1.0, 1.4);

// Converts Carla RGB Camera image to OpenCV image, by popping the oldest element in the queue
void CarlaRGBToOpenCV(TSQueue<boost::shared_ptr<csd::Image>> &carla_image_queue, cv::Mat &cv_img)
{
    boost::shared_ptr<csd::Image> carla_img_ptr;
    carla_image_queue.dequeue(carla_img_ptr);

    csd::Color *image_data = carla_img_ptr->data();

    for (int ii = 0; ii < IMAGE_HEIGHT; ii++)
    {
        for (int jj = 0; jj < IMAGE_WIDTH; jj++)
        {
            csd::Color color    = image_data[jj + ii * IMAGE_WIDTH];
            cv::Vec3b &cv_color = cv_img.at<cv::Vec3b>(ii, jj);
            cv_color[0]         = color.b;
            cv_color[1]         = color.g;
            cv_color[2]         = color.r;
        }
    }
}

// Converts carla depth image to Opencv 1 channel depth image, by popping the oldest element in the queue
void CarlaDepthToOpenCV(TSQueue<boost::shared_ptr<csd::Image>> &carla_image_queue, cv::Mat &cv_depth_img_F32)
{
    boost::shared_ptr<csd::Image> carla_img_ptr;
    carla_image_queue.dequeue(carla_img_ptr);

    csd::Color *image_data = carla_img_ptr->data();

    for (int ii = 0; ii < IMAGE_HEIGHT; ii++)
    {
        for (int jj = 0; jj < IMAGE_WIDTH; jj++)
        {
            csd::Color color = image_data[jj + ii * IMAGE_WIDTH];

            // A) Leave calculating depth to the user
            // cv::Vec3b &cv_color = cv_depth_img_F32.at<cv::Vec3b>(ii, jj);
            // cv_color[0]         = color.b;
            // cv_color[1]         = color.g;
            // cv_color[2]         = color.r;

            // B) Calculate the real depth here
            float &depth_at_pixel = cv_depth_img_F32.at<float>(ii, jj);
            depth_at_pixel        = static_cast<float>(color.r + color.g * 256 + color.b * 256 * 256) /
                             static_cast<float>(256 * 256 * 256 - 1) * 1000.0;
        }
    }
}

cv::Mat getCameraIntrinsic(const carla::client::BlueprintLibrary::value_type &camera_bp)
{
    // Eigen::Matrix3f K = Eigen::Matrix3f::Identity(); // intrinsics matrix for pinhole
    int   image_w = camera_bp.GetAttribute("image_size_x").As<int>();
    int   image_h = camera_bp.GetAttribute("image_size_y").As<int>();
    float fov     = camera_bp.GetAttribute("fov").As<float>();
    float focal   = static_cast<float>(image_w) / (2.0f * std::tan(fov * M_PI / 360.0f));
    float c_x     = static_cast<float>(image_w) / 2.0;
    float c_y     = static_cast<float>(image_h) / 2.0;

    cv::Mat K = (cv::Mat_<float>(3, 3) << focal, 0.0, c_x, 0.0, focal, c_y, 0.0, 0.0, 1.0);

    std::cout << "Camera Instrinsics:" << std::endl;
    std::cout << K << std::endl;
    return K;
}

int main()
{
    std::string host("localhost");
    uint16_t    port(2000u);

    auto client = cc::Client(host, port);
    client.SetTimeout(40s);

    std::cout << "Client API version : " << client.GetClientVersion() << '\n';
    std::cout << "Server API version : " << client.GetServerVersion() << '\n';

    cc::World world           = client.GetWorld();
    auto      settings        = world.GetSettings();
    auto      traffic_manager = client.GetInstanceTM();
    traffic_manager.SetSynchronousMode(true);
    settings.fixed_delta_seconds = SIM_STEP_TIME;
    settings.synchronous_mode    = true;
    settings.no_rendering_mode   = true;
    world.ApplySettings(settings, 3s);

    // Get a random vehicle blueprint.
    auto blueprint_library = world.GetBlueprintLibrary();

    // Find a valid spawn point.
    auto map         = world.GetMap(); // Assumes Town03
    auto spawnPoints = map->GetRecommendedSpawnPoints();
    for (auto &spawnPt : spawnPoints)
        spawnPt.location.z += 5.0;

    // Find a camera blueprint & configure cam
    auto camera_rgb_bp_left  = (*(blueprint_library->Filter("sensor.camera.rgb")))[0];
    auto camera_rgb_bp_right = (*(blueprint_library->Filter("sensor.camera.rgb")))[0];
    auto camera_depth_bp     = (*(blueprint_library->Filter("sensor.camera.depth")))[0];
    camera_rgb_bp_left.SetAttribute("image_size_x", std::to_string(IMAGE_WIDTH));
    camera_rgb_bp_left.SetAttribute("image_size_y", std::to_string(IMAGE_HEIGHT));
    camera_rgb_bp_right.SetAttribute("image_size_x", std::to_string(IMAGE_WIDTH));
    camera_rgb_bp_right.SetAttribute("image_size_y", std::to_string(IMAGE_HEIGHT));
    camera_depth_bp.SetAttribute("image_size_x", std::to_string(IMAGE_WIDTH));
    camera_depth_bp.SetAttribute("image_size_y", std::to_string(IMAGE_HEIGHT));

    // Attach the left camera and right camera to depth camera, left and depth are colocated
    // (flip y since UE4 left handed)
    auto left_cam_mounting =
        cg::Transform{cg::Location{CAMERA_LEFT_POS_X, -CAMERA_LEFT_POS_Y, CAMERA_LEFT_POS_Z}, // x, y, z.
                      cg::Rotation{0.0f, 0.0f, 0.0f}};                                        // pitch, yaw, roll.
    auto right_cam_mounting =
        cg::Transform{cg::Location{CAMERA_RIGHT_POS_X, -CAMERA_RIGHT_POS_Y, CAMERA_RIGHT_POS_Z}, // x, y, z.
                      cg::Rotation{0.0f, 0.0f, 0.0f}};                                           // pitch, yaw, roll.
    auto depth_camera_actor     = world.SpawnActor(camera_depth_bp, spawnPoints[0]);
    auto depth_camera           = boost::static_pointer_cast<cc::Sensor>(depth_camera_actor);
    auto rgb_camera_actor_left  = world.SpawnActor(camera_rgb_bp_left, left_cam_mounting, depth_camera_actor.get());
    auto rgb_camera_actor_right = world.SpawnActor(camera_rgb_bp_right, right_cam_mounting, depth_camera_actor.get());
    auto rgb_camera_left        = boost::static_pointer_cast<cc::Sensor>(rgb_camera_actor_left);
    auto rgb_camera_right       = boost::static_pointer_cast<cc::Sensor>(rgb_camera_actor_right);

    // Thread-safe queue for storing the captured images
    TSQueue<boost::shared_ptr<csd::Image>> carla_rgb_image_queue_left, carla_rgb_image_queue_right,
        carla_depth_image_queue;
    rgb_camera_left->Listen(
        [&](auto data)
        {
            boost::shared_ptr<csd::Image> image_ptr = boost::static_pointer_cast<csd::Image>(data);
            carla_rgb_image_queue_left.enqueue(image_ptr);
        });
    rgb_camera_right->Listen(
        [&](auto data)
        {
            boost::shared_ptr<csd::Image> image_ptr = boost::static_pointer_cast<csd::Image>(data);
            carla_rgb_image_queue_right.enqueue(image_ptr);
        });

    depth_camera->Listen(
        [&](auto data)
        {
            boost::shared_ptr<csd::Image> image_ptr = boost::static_pointer_cast<csd::Image>(data);
            carla_depth_image_queue.enqueue(image_ptr);
        });

    // Stereo depth
    // Extrinsics are specified in OpenCV convention: X: right, Y: down, Z: outwards from camera
    Eigen::Matrix4f extrinsics_left_to_right_cam(Eigen::Matrix4f::Identity());
    extrinsics_left_to_right_cam(0, 3) = -CAMERA_RIGHT_POS_Y; // x: right
    extrinsics_left_to_right_cam(1, 3) = 0;                   // y: down
    extrinsics_left_to_right_cam(2, 3) = 0;                   // z: up
    StereoDepth stereo_depth(
        getCameraIntrinsic(camera_rgb_bp_left), getCameraIntrinsic(camera_rgb_bp_right), extrinsics_left_to_right_cam);

    cv::Mat cv_img_left(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat cv_img_right(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat cv_depth_img_F32(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F, cv::Scalar(0.0));
    bool    init_frame = true;

    auto        pos_itr = spawnPoints.begin();
    std::string line;
    while (std::getline(std::cin, line)) // until space is pressed
    {
        std::cout << "changed view" << std::endl;
        if (pos_itr != spawnPoints.end())
        {
            pos_itr++;
            depth_camera->SetTransform(*pos_itr);
        }
        else
        {
            pos_itr = spawnPoints.begin();
        }

        auto frame_id = world.Tick(1s); // timeout
        std::this_thread::sleep_for(5ms);

        CarlaRGBToOpenCV(carla_rgb_image_queue_left, cv_img_left);
        CarlaRGBToOpenCV(carla_rgb_image_queue_right, cv_img_right);
        CarlaDepthToOpenCV(carla_depth_image_queue, cv_depth_img_F32);
        cv::imshow("left", cv_img_left);
        cv::imshow("right", cv_img_right);

        // auto disparityLeftBM   = stereo_depth.computeLeftDisparityMapBM(cv_img_left, cv_img_right);
        auto disparityLeftSGBM = stereo_depth.computeLeftDisparityMapSGBM(cv_img_left, cv_img_right);
        auto depthMap          = stereo_depth.computeDepthFromLeftDisparityMap(disparityLeftSGBM);
        stereo_depth.projectLeftImgTo3D(cv_img_left, depthMap);
        // stereo_depth.projectLeftImgTo3D(cv_img_left, cv_depth_img_F32);

        cv::waitKey(33);
    }
    std::cout << "done" << std::endl;

    // rgb_camera->SetTransform(*pos_itr);
    // depth_camera->SetTransform(*pos_itr);
    // pos_itr++;

    return 0;
}