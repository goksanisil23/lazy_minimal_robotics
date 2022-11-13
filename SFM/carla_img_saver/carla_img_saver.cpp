#include <algorithm>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
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

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv4/opencv2/opencv.hpp>

#include "ThreadSafeQueue.h"

#include <SFML/Window/Keyboard.hpp>

namespace cc  = carla::client;
namespace cg  = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

constexpr int16_t   IMAGE_WIDTH   = 1024;
constexpr int16_t   IMAGE_HEIGHT  = 640;
constexpr u_int32_t NUM_SIM_STEPS = 1000;
constexpr double    SIM_STEP_TIME = 0.1; // secs

constexpr double NEXT_WAYPOINT_DIST = 0.5;
constexpr int    NUM_WAYPOINTS      = 50;

const std::string RGB_SAVE_PREFIX   = "../../resources/data/imgs/rgb/rgb_";
const std::string DEPTH_SAVE_PREFIX = "../../resources/data/imgs/depth/depth_";
const std::string CAM_POSE_TXT_PATH = "../../resources/data/camera_poses_gt.txt";

static bool  is_reverse = false;
static float throttle   = 0.0;
static float steer      = 0.0;
static float brake      = 0.0;

std::mutex              controlsMtx;
std::condition_variable controlsCondVar;

struct RoboticsPose
{
    float x, y, z;
    float qx, qy, qz, qw;
};

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
void CarlaDepthToOpenCV(TSQueue<boost::shared_ptr<csd::Image>> &carla_image_queue, cv::Mat &cv_img)
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
            cv::Vec3b &cv_color = cv_img.at<cv::Vec3b>(ii, jj);
            cv_color[0]         = color.b;
            cv_color[1]         = color.g;
            cv_color[2]         = color.r;

            // B) Calculate the real depth here
            // double &depth_at_pixel = cv_img.at<double>(ii, jj);
            // depth_at_pixel = static_cast<double>(color.r + color.g * 256 + color.b * 256 * 256) / static_cast<double>(256 * 256 * 256 - 1) * 1000.0;
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

void updateControls()
{
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        std::lock_guard<std::mutex> lock{controlsMtx};
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
        {
            throttle += 0.01f;
            throttle = std::clamp(throttle, 0.0f, 1.0f);
        }
        else
        {
            throttle -= 0.01f;
            throttle = std::clamp(throttle, 0.0f, 1.0f);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
        {
            brake += 0.2f;
            brake = std::clamp(brake, 0.0f, 1.0f);
        }
        else
        {
            brake -= 0.2f;
            brake = std::clamp(brake, 0.0f, 1.0f);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
        {
            steer -= 0.01f;
            steer = std::clamp(steer, -1.0f, 1.0f);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
        {
            steer += 0.01f;
            steer = std::clamp(steer, -1.0f, 1.0f);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Q))
        {
            is_reverse = !is_reverse;
        }
    }

    // controlsCondVar.notify_one();
}

std::string zeroPadNumber(int number)
{
    static const size_t padLen       = 5;
    std::string         numberStr    = std::to_string(number);
    auto                paddedNumStr = std::string(padLen - std::min(padLen, numberStr.length()), '0') + numberStr;
    return paddedNumStr;
}

void applyManualControl(boost::shared_ptr<cc::Vehicle> vehicle)
{
    std::lock_guard<std::mutex> lock{controlsMtx};
    std::cout << "t: " << throttle << " b: " << brake << " s: " << steer << " reverse:" << is_reverse << std::endl;
    vehicle->ApplyControl(cc::Vehicle::Control(throttle, steer, brake, false, is_reverse, false, 0));
}

void showImage(const cv::Mat &cv_img)
{
    static int imgCtr = 0;
    while (true)
    {
        // std::this_thread::sleep_for(std::chrono::milliseconds(33));
        // std::lock_guard<std::mutex> lock{controlsMtx};
        cv::imshow("img", cv_img);
        cv::waitKey(33);
        std::cout << "imgctr: " << imgCtr << std::endl;
        imgCtr++;
    }
}

void carlaTransformToRoboticsConvention(const cg::Transform &camera_pose, RoboticsPose &camera_pose_robotics)
{
    camera_pose_robotics.x = camera_pose.location.x;
    camera_pose_robotics.y = camera_pose.location.y;
    camera_pose_robotics.z = camera_pose.location.z;

    float roll  = camera_pose.rotation.roll;
    float pitch = -camera_pose.rotation.pitch;
    float yaw   = -camera_pose.rotation.yaw;

    Eigen::Quaternionf q;
    q = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());

    camera_pose_robotics.qx = q.coeffs().x();
    camera_pose_robotics.qy = q.coeffs().y();
    camera_pose_robotics.qz = q.coeffs().z();
    camera_pose_robotics.qw = q.coeffs().w();
}

void setTextureForSculpture(cc::World &world)
{
    cv::Mat textureCvImg = cv::imread("../../resources/texture_for_carla/texture.png");
    std::cout << "read texture" << std::endl;
    auto texture_height = textureCvImg.rows;
    auto texture_width  = textureCvImg.cols;

    auto carla_texture = carla::rpc::TextureColor(texture_width, texture_height);

    for (int ii = 0; ii < texture_height; ii++)
    {
        for (int jj = 0; jj < texture_width; jj++)
        {
            csd::Color carla_color;

            // A) Leave calculating depth to the user
            cv::Vec3b &cv_color = textureCvImg.at<cv::Vec3b>(ii, jj);
            carla_color.r       = cv_color[0];
            carla_color.g       = cv_color[1];
            carla_color.b       = cv_color[2];
            carla_color.a       = 255;

            carla_texture.At(jj, ii) = carla_color;
        }
    }

    world.ApplyColorTextureToObject(
        "Sculpture_StaticMesh2_33", carla::rpc::MaterialParameter::Tex_Diffuse, carla_texture);
}

int main()
{
    try
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
        auto map = world.GetMap(); // Assumes Town03
        // Change the texture of the roundabout architecture
        setTextureForSculpture(world);

        auto spawn_transform  = cg::Transform{cg::Location{15.4, 9.2, 0.6}, cg::Rotation{10.0f, 0.0f, 0.0f}};
        auto current_waypoint = world.GetMap()->GetWaypoint(spawn_transform.location);
        std::vector<cg::Transform> position_list;
        for (int i = 0; i < NUM_WAYPOINTS; i++)
        {
            auto next_waypoint = current_waypoint->GetNext(NEXT_WAYPOINT_DIST).at(0);
            auto wp            = next_waypoint->GetTransform();
            wp.rotation.yaw += -90.0;
            wp.location.z += 4.0;
            wp.rotation.pitch -= 5.0;
            position_list.push_back(wp);
            auto pt = next_waypoint->GetTransform().location;
            world.MakeDebugHelper().DrawPoint(pt, 0.1, cc::Color(0, 0, 255, 255), 0, false);
            current_waypoint = next_waypoint;
        }

        // Find a camera blueprint.
        auto camera_rgb_bp   = (*(blueprint_library->Filter("sensor.camera.rgb")))[0];
        auto camera_depth_bp = (*(blueprint_library->Filter("sensor.camera.depth")))[0];
        // Spawn a camera attached to the vehicle.
        auto camera_transform = cg::Transform{*(position_list.begin())};

        camera_rgb_bp.SetAttribute("image_size_x", std::to_string(IMAGE_WIDTH));
        camera_rgb_bp.SetAttribute("image_size_y", std::to_string(IMAGE_HEIGHT));
        camera_depth_bp.SetAttribute("image_size_x", std::to_string(IMAGE_WIDTH));
        camera_depth_bp.SetAttribute("image_size_y", std::to_string(IMAGE_HEIGHT));

        getCameraIntrinsic(camera_rgb_bp);

        auto rgb_camera_actor   = world.SpawnActor(camera_rgb_bp, camera_transform);
        auto rgb_camera         = boost::static_pointer_cast<cc::Sensor>(rgb_camera_actor);
        auto depth_camera_actor = world.SpawnActor(camera_depth_bp, camera_transform);
        auto depth_camera       = boost::static_pointer_cast<cc::Sensor>(depth_camera_actor);

        // Thread-safe queue for storing the captured images
        TSQueue<boost::shared_ptr<csd::Image>> carla_rgb_image_queue;
        rgb_camera->Listen(
            [&](auto data)
            {
                boost::shared_ptr<csd::Image> image_ptr = boost::static_pointer_cast<csd::Image>(data);
                carla_rgb_image_queue.enqueue(image_ptr);
            });

        TSQueue<boost::shared_ptr<csd::Image>> carla_depth_image_queue;
        depth_camera->Listen(
            [&](auto data)
            {
                boost::shared_ptr<csd::Image> image_ptr = boost::static_pointer_cast<csd::Image>(data);
                carla_depth_image_queue.enqueue(image_ptr);
            });

        cv::Mat cv_img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
        // cv::Mat cv_depth_img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_64F, cv::Scalar(0.0));
        cv::Mat       cv_depth_img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, cv::Scalar(0.0));
        cg::Transform camera_origo;
        cg::Transform camera_odom;
        bool          init_frame = true;

        // Start the keyboard handling thread for manual controls
        // std::thread keyboardThread(updateControls);
        // std::thread renderThread(showImage, cv_img);

        auto pos_itr = position_list.begin();
        rgb_camera->SetTransform(*pos_itr);
        depth_camera->SetTransform(*pos_itr);
        pos_itr++;

        // Advance the simulation
        int ii = 0;

        std::ofstream cam_pose_out_txt;
        cam_pose_out_txt.open(CAM_POSE_TXT_PATH);

        while (true)
        {
            ii++;
            auto frame_id = world.Tick(1s); // timeout
            std::this_thread::sleep_for(5ms);

            CarlaRGBToOpenCV(carla_rgb_image_queue, cv_img);
            CarlaDepthToOpenCV(carla_depth_image_queue, cv_depth_img);
            cv::imshow("img", cv_img);
            cv::waitKey(33);

            std::cout << "ii: " << ii << std::endl;

            cg::Transform camera_pose = rgb_camera->GetTransform();
            // Visualize
            RoboticsPose camera_pose_robotics;
            carlaTransformToRoboticsConvention(camera_pose, camera_pose_robotics);
            cg::Vector3D camera_position(camera_pose.location.x, camera_pose.location.y, camera_pose.location.z);
            cam_pose_out_txt << ii << " " << camera_pose_robotics.x << " " << camera_pose_robotics.y << " "
                             << camera_pose_robotics.z << " " << camera_pose_robotics.qx << " "
                             << camera_pose_robotics.qy << " " << camera_pose_robotics.qz << " "
                             << camera_pose_robotics.qw << "\n";

            int         imgNum        = std::distance(position_list.begin(), pos_itr);
            std::string numStr        = zeroPadNumber(imgNum);
            std::string rgbSavePath   = RGB_SAVE_PREFIX + numStr + std::string(".png");
            std::string depthSavePath = DEPTH_SAVE_PREFIX + numStr + std::string(".png");
            cv::imwrite(rgbSavePath, cv_img);
            cv::imwrite(depthSavePath, cv_depth_img);

            rgb_camera->SetTransform(*pos_itr);
            depth_camera->SetTransform(*pos_itr);
            if (pos_itr != position_list.end())
            {
                pos_itr++;
            }
            else
            {
                break;
            }
        }

        // Remove actors from the simulation.
        rgb_camera->Destroy();
        depth_camera->Destroy();
        cam_pose_out_txt.close();
    }
    catch (const cc::TimeoutException &e)
    {
        std::cout << '\n' << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cout << "\nException: " << e.what() << std::endl;
        return 2;
    }
}
