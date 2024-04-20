#include "raylib-cpp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "Agent.h"
#include "GraphSlam.h"
#include "Landmark.h"
#include "Odometry.h"
#include "Visualizer.h"

constexpr size_t kNumLandmarks{75};
constexpr float  kMinLandmarkSpacing{10.F};

struct DeadReckon
{
    raylib::Vector2 position;
    float           heading;

    // Accumlates delta_pose from robot frame into global frame
    void update(const raylib::Vector2 delta_pos, const float delta_rot)
    {
        float global_dx = delta_pos.x * cos(heading) - delta_pos.y * sin(heading);
        float global_dy = delta_pos.x * sin(heading) + delta_pos.y * cos(heading);

        // Update the global pose
        position.x += global_dx;
        position.y += global_dy;
        heading += delta_rot;

        // Normalize the angle to remain within -PI to PI
        heading = fmod(heading + M_PI, 2 * M_PI);
        if (heading < 0)
            heading += 2 * M_PI;
        heading -= M_PI;
    }
};

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

int main()
{
    // Agent agent{{0, 0}, M_PI / 4};
    Agent                 agent{{Visualizer::kScreenWidth / 2, Visualizer::kScreenHeight / 2}, M_PI / 4};
    DeadReckon            dead_reckon{{agent.position_.x, agent.position_.y}, agent.heading_};
    Odometry              odometry;
    raylib::Vector2       opt_pose{agent.position_.x, agent.position_.y};
    std::vector<Landmark> landmarks =
        generateLandmarks(kNumLandmarks, Visualizer::kScreenWidth, Visualizer::kScreenHeight, kMinLandmarkSpacing);

    Visualizer visualizer;

    GraphSlam graph_slam(agent.position_.x, agent.position_.y, agent.heading_);

    while (!visualizer.shouldClose())
    {
        float linear_vel = 0.0, angular_vel = 0.0;

        if (IsKeyDown(KEY_UP))
            linear_vel = 80.0;
        if (IsKeyDown(KEY_DOWN))
            linear_vel = -80.0;
        if (IsKeyDown(KEY_RIGHT))
            angular_vel = 1.;
        if (IsKeyDown(KEY_LEFT))
            angular_vel = -1.;

        auto [linear_vel_meas, angular_vel_meas] = agent.readIMU(linear_vel, angular_vel);
        agent.updateMovement(linear_vel, angular_vel);
        odometry.update(linear_vel_meas, angular_vel_meas);
        dead_reckon.update(odometry.delta_position_, odometry.delta_heading_);

        agent.senseLandmarks(landmarks);

        graph_slam.processMeasurements(odometry.delta_position_, odometry.delta_heading_, agent.landmark_measurements_);

        opt_pose = graph_slam.getLastOptPose();

        std::cout << "GT: " << agent.position_.x << " " << agent.position_.y << std::endl;
        std::cout << "SL: " << opt_pose.x << " " << opt_pose.y << std::endl;

        visualizer.draw(agent, landmarks, dead_reckon.position, opt_pose, graph_slam.landmarks_);
    }
    return 0;
}
