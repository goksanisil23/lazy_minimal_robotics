#pragma once

#include <vector>

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

namespace cc  = carla::client;
namespace cg  = carla::geom;
namespace csd = carla::sensor::data;

std::vector<cg::Transform> GenerateSfmPoseList(const cc::World &world)
{
    constexpr double NEXT_WAYPOINT_DIST = 0.5;
    constexpr int    NUM_WAYPOINTS      = 50;

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

    return position_list;
}

std::vector<cg::Transform> GeneratePureRotationPoseList(const cc::World &world)
{
    constexpr double ROTATION_STEP_DEG = 10.0;
    constexpr int    NUM_ROTATIONS     = 6;

    auto spawn_transform  = cg::Transform{cg::Location{15.4, 9.2, 0.6}, cg::Rotation{10.0f, 0.0f, 0.0f}};
    auto current_waypoint = world.GetMap()->GetWaypoint(spawn_transform.location);
    std::vector<cg::Transform> position_list;
    for (int i = 0; i < NUM_ROTATIONS; i++)
    {
        auto next_pose = current_waypoint->GetTransform();
        next_pose.rotation.yaw += -130.0;
        next_pose.location.z += 4.0;
        next_pose.rotation.pitch -= 5.0;

        next_pose.rotation.yaw += i * ROTATION_STEP_DEG;
        position_list.push_back(next_pose);
        world.MakeDebugHelper().DrawPoint(next_pose.location, 0.1, cc::Color(0, 0, 255, 255), 0, false);
    }

    return position_list;
}