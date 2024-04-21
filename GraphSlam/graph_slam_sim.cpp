#include "raylib-cpp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "Agent.h"
#include "DeadReckon.h"
#include "GraphSlam.h"
#include "Landmark.h"
#include "Odometry.h"
#include "Visualizer.h"

constexpr size_t kNumLandmarks{150};
constexpr float  kMinLandmarkSpacing{10.F};

std::vector<Landmark> generateLandmarks(size_t count, int area_width, int area_height, float min_distance)
{
    std::vector<Landmark>                 landmarks;
    std::default_random_engine            gen;
    std::uniform_real_distribution<float> dist_x(0, area_width);
    std::uniform_real_distribution<float> dist_y(0, area_height);

    while (landmarks.size() < count)
    {
        Landmark new_landmark{{dist_x(gen), dist_y(gen)}, static_cast<int>(landmarks.size())};
        bool     too_close =
            std::any_of(landmarks.begin(),
                        landmarks.end(),
                        [&](const Landmark &l) { return new_landmark.position_.Distance(l.position_) < min_distance; });

        if (!too_close)
        {
            std::cout << "Added landmark " << new_landmark.id_ << std::endl;
            landmarks.push_back(new_landmark);
        }
    }
    return landmarks;
}

void manualControl(float &linear_vel, float &angular_vel)
{
    if (IsKeyDown(KEY_UP))
        linear_vel = 80.0;
    if (IsKeyDown(KEY_DOWN))
        linear_vel = -80.0;
    if (IsKeyDown(KEY_RIGHT))
        angular_vel = 1.;
    if (IsKeyDown(KEY_LEFT))
        angular_vel = -1.;
}

int main()
{
    Agent      agent{{Visualizer::kScreenWidth / 2, Visualizer::kScreenHeight / 2}, M_PI / 4};
    DeadReckon dead_reckon{{agent.position_.x, agent.position_.y}, agent.heading_};
    Odometry   odometry;

    std::vector<Landmark> landmarks =
        generateLandmarks(kNumLandmarks, Visualizer::kScreenWidth, Visualizer::kScreenHeight, kMinLandmarkSpacing);

    Visualizer visualizer;

    GraphSlam graph_slam(agent.position_.x, agent.position_.y, agent.heading_);

    while (!visualizer.shouldClose())
    {
        float linear_vel = 0.0, angular_vel = 0.0;
        manualControl(linear_vel, angular_vel);

        auto [linear_vel_meas, angular_vel_meas] = agent.readIMU(linear_vel, angular_vel);
        agent.updateMovement(linear_vel, angular_vel);
        odometry.update(linear_vel_meas, angular_vel_meas);
        dead_reckon.update(odometry.delta_position_, odometry.delta_heading_);

        agent.senseLandmarks(landmarks);

        graph_slam.processMeasurements(odometry.delta_position_, odometry.delta_heading_, agent.landmark_measurements_);

        auto opt_pose = graph_slam.getLastOptPose();

        // std::cout << "GT: " << agent.position_.x << " " << agent.position_.y << std::endl;
        // std::cout << "SL: " << opt_pose.x << " " << opt_pose.y << std::endl;

        visualizer.draw(agent, landmarks, dead_reckon, opt_pose, graph_slam.landmarks_);
    }
    return 0;
}
