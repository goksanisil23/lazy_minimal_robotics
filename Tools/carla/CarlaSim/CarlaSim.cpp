#include "CarlaSim.h"

void CarlaSim::Setup()
{
    std::string host("localhost");
    uint16_t    port(2000u);

    client_ = std::make_shared<cc::Client>(host, port);
    client_->SetTimeout(40s);

    std::cout << "Client API version : " << client_->GetClientVersion() << '\n';
    std::cout << "Server API version : " << client_->GetServerVersion() << '\n';

    world_               = std::make_shared<cc::World>(client_->GetWorld());
    auto settings        = world_->GetSettings();
    originalSettings_    = settings;
    auto traffic_manager = client_->GetInstanceTM();
    traffic_manager.SetSynchronousMode(true);
    settings.fixed_delta_seconds = SIM_STEP_TIME;
    settings.synchronous_mode    = true;
    world_->ApplySettings(settings, 3s);

    // Create spectator so we can see the vehicle from the simulator window.
    spectator_ = world_->GetSpectator();

    SetupCloudViz();
}

boost::shared_ptr<cc::Vehicle> CarlaSim::SpawnVehicle()
{
    // Get a random vehicle blueprint.
    blueprintLib_   = world_->GetBlueprintLibrary();
    auto vehicle_bp = blueprintLib_->Find("vehicle.volkswagen.t2");
    // // auto blueprint = RandomChoice(*vehicles, rng);

    // // Find a valid spawn point.
    auto map             = world_->GetMap();
    auto spawnTransform  = map->GetRecommendedSpawnPoints()[0];
    auto currentWaypoint = world_->GetMap()->GetWaypoint(spawnTransform.location);

    // // Spawn the vehicle.
    auto actor = world_->SpawnActor(*vehicle_bp, spawnTransform);
    std::cout << "Spawned " << actor->GetDisplayId() << '\n';
    vehicle_ = boost::static_pointer_cast<cc::Vehicle>(actor);

    vehicle_->SetAutopilot(true);

    // boost::shared_ptr<cc::Vehicle> vehiclePtr{vehicle_};
    // return vehiclePtr;
    return vehicle_;
}

boost::shared_ptr<cc::Sensor> CarlaSim::SpawnLidar()
{
    // Find a lidar blueprint.
    // auto lidarBp = (*(blueprint_library->Filter("sensor.lidar.ray_cast")))[0];
    auto lidarBp = (*(blueprintLib_->Filter("sensor.lidar.ray_cast_semantic")))[0];
    // Spawn a lidar attached to the vehicle.
    auto lidarTransform = cg::Transform{cg::Location{0.f, 0.0f, 2.8f},   // x, y, z.
                                        cg::Rotation{0.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.

    // auto &attribute = lidarBp->GetAttribute("upper_fov");
    lidarBp.SetAttribute("upper_fov", "22.5");

    // lidarBp.SetAttribute("dropoff_general_rate", "0.0");
    // lidarBp.SetAttribute("dropoff_intensity_limit", "1.0");
    // lidarBp.SetAttribute("dropoff_zero_intensity", "0.0");
    lidarBp.SetAttribute("upper_fov", "22.5");
    lidarBp.SetAttribute("lower_fov", "-22.5");
    lidarBp.SetAttribute("channels", "64");
    lidarBp.SetAttribute("range", "150.0");
    lidarBp.SetAttribute("points_per_second", "327680");
    lidarBp.SetAttribute("rotation_frequency", "10");
    // lidarBp.SetAttribute("sensor_tick", "0.1");
    auto lidarActor = world_->SpawnActor(lidarBp, lidarTransform, vehicle_.get());
    lidar_          = boost::static_pointer_cast<cc::Sensor>(lidarActor);

    // boost::shared_ptr<cc::Sensor> lidarPtr{lidar_};
    // return lidarPtr;
    return lidar_;
}

// Move the server-side spectator according to the vehicle position
void CarlaSim::MoveSpectator()
{
    // Update the camera
    auto spectatorTransform = vehicle_->GetTransform();
    spectatorTransform.location.z += 3.0;
    float dx = cos(cg::Math::ToRadians(spectatorTransform.rotation.yaw)) * 5.0;
    float dy = sin(cg::Math::ToRadians(spectatorTransform.rotation.yaw)) * 5.0;
    spectatorTransform.location.x -= dx;
    spectatorTransform.location.y -= dy;
    spectator_->SetTransform(spectatorTransform);
}

uint64_t CarlaSim::Step()
{
    return world_->Tick(1s); // timeout
}

void CarlaSim::LidarCallback(boost::shared_ptr<carla::sensor::SensorData> lidarDataPtr)
{
    // auto pointcloud = boost::static_pointer_cast<csd::Array<csd::LidarDetection>>(lidar_data);
    auto pointcloudPtr = boost::static_pointer_cast<SemanticLidarData>(lidarDataPtr);
    lidarDataPtrQueue_.Enqueue(pointcloudPtr);
};

// Pops the front element in the lidar queue (the oldest). Blocking if the queue is empty until its not.
void CarlaSim::GetLidarData(boost::shared_ptr<SemanticLidarData> &lidarDataPtr)
{
    lidarDataPtrQueue_.Dequeue(lidarDataPtr);
}

void CarlaSim::GetLidarPose(Eigen::Matrix4f &lidarPose)
{
    CarlaToRoboticsTransform(lidar_->GetTransform(), lidarPose);
}

void CarlaSim::Terminate()
{
    // Remove actors from the simulation.
    if (lidar_)
        lidar_->Destroy();
    if (vehicle_)
        vehicle_->Destroy();
    // Set it back to async
    if (world_)
        world_->ApplySettings(originalSettings_, 3s);
}

void CarlaSim::SetupCloudViz()
{
    // setup visualization
    o3dCloudVis_.CreateVisualizerWindow("Lidar data", 960, 540, 480, 270);
    o3dCloudVis_.GetRenderOption().background_color_      = {0.05, 0.05, 0.05};
    o3dCloudVis_.GetRenderOption().point_size_            = 1;
    o3dCloudVis_.GetRenderOption().show_coordinate_frame_ = true;

    // Create initial cloud to determine the bounding box of the viewer
    o3dPoints_.push_back({-100.0, -100.0, 1.0});
    o3dPoints_.push_back({-100.0, 100.0, 2.0});
    o3dPoints_.push_back({100.0, 100.0, 3.0});
    o3dPoints_.push_back({100.0, -100.0, 4.0});
    o3dCloud_ = std::make_shared<open3d::geometry::PointCloud>(o3dPoints_);
    o3dCloudVis_.AddGeometry(o3dCloud_);
}
void CarlaSim::UpdateCloudViz(boost::shared_ptr<SemanticLidarData> pointcloudPtr)
{
    // Create Open3D pointcloud from carla pointcloud
    o3dPoints_.clear();
    o3dColors_.clear();
    for (auto &pt : *pointcloudPtr)
    {
        o3dPoints_.push_back({pt.point.x, pt.point.y, pt.point.z});
        o3dColors_.emplace_back(semSegColors_.GetColorForId(pt.object_tag).data());
    }
    o3dCloud_->points_ = o3dPoints_;
    o3dCloud_->colors_ = o3dColors_;
    o3dCloudVis_.UpdateGeometry(o3dCloud_);
    o3dCloudVis_.PollEvents();
    o3dCloudVis_.UpdateRender();
}

void CarlaSim::CarlaToRoboticsTransform(const cg::Transform &carlaPose, Eigen::Matrix4f &roboticsPose)
{
    roboticsPose.setIdentity();

    roboticsPose.block<3, 1>(0, 3) = Eigen::Vector3f{carlaPose.location.x, -carlaPose.location.y, carlaPose.location.z};

    float              roll  = cg::Math::ToRadians(carlaPose.rotation.roll);
    float              pitch = cg::Math::ToRadians(-carlaPose.rotation.pitch);
    float              yaw   = cg::Math::ToRadians(-carlaPose.rotation.yaw);
    Eigen::Quaternionf q;
    q = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
        Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
    roboticsPose.block<3, 3>(0, 0) = q.matrix();
}
