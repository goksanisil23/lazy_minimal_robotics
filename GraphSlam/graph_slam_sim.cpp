#include "raylib-cpp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

constexpr float kDt = 0.015;

struct Landmark
{
    raylib::Vector2 position_;
    int             id_;
};

std::vector<Landmark> generateLandmarks(size_t count, int area_width, int area_height, float min_distance)
{
    std::vector<Landmark>                 landmarks;
    std::default_random_engine            gen;
    std::uniform_real_distribution<float> dist_x(0, area_width);
    std::uniform_real_distribution<float> dist_y(0, area_height);

    while (landmarks.size() < count)
    {
        Landmark new_landmark{{dist_x(gen), dist_y(gen)}};
        bool     too_close =
            std::any_of(landmarks.begin(),
                        landmarks.end(),
                        [&](const Landmark &l) { return new_landmark.position_.Distance(l.position_) < min_distance; });

        if (!too_close)
        {
            landmarks.push_back(new_landmark);
        }
    }
    return landmarks;
}

class Agent
{
  public:
    static constexpr float kRadius{10.F};
    static constexpr float kSensorRange{100.F};

    struct LandmarkDetection
    {
        float range;
        float bearing;
        int   id;
    };

    Agent(const raylib::Vector2 position_initial, const float angle_initial)
        : position_{position_initial}, angle_{angle_initial}
    {
    }

    raylib::Vector2 position_;
    float           angle_ = 0.0; // Initial heading angle_ in radians

    std::vector<LandmarkDetection> landmark_detections_{};

    std::default_random_engine      noise_generator;
    std::normal_distribution<float> linear_vel_noise_dist{0.0, 5.0};       // Linear velocity noise
    std::normal_distribution<float> angular_vel_noise_dist{0.0, 0.2};      // Angular velocity noise
    std::normal_distribution<float> landmark_range_noise_dist{0.0, 2.0};   // Sensor noise for distance
    std::normal_distribution<float> landmark_bearing_noise_dist{0.0, 0.5}; // Sensor noise for bearing

    // IMU simulated measurements
    std::pair<float, float> readIMU(float linearVelocity, float angularVelocity)
    {
        return {
            linearVelocity + linear_vel_noise_dist(noise_generator),  // Noisy linear velocity
            angularVelocity + angular_vel_noise_dist(noise_generator) // Noisy angular velocity
        };
    }

    // Update movement based on actual commands without noise
    void updateMovement(float linearVelocity, float angularVelocity)
    {
        angle_ += angularVelocity * kDt;
        position_.x += linearVelocity * kDt * cos(angle_);
        position_.y += linearVelocity * kDt * sin(angle_);
    }

    // Sensing landmarks
    void senseLandmarks(const std::vector<Landmark> &landmarks)
    {
        landmark_detections_.clear();

        for (const auto &landmark : landmarks)
        {
            float distance = position_.Distance(landmark.position_);
            float bearing  = atan2(landmark.position_.y - position_.y, landmark.position_.x - position_.x) - angle_;
            if (distance <= kSensorRange)
            {
                // Simulate measurement noise for distance and bearing
                distance += landmark_range_noise_dist(noise_generator);
                bearing += landmark_bearing_noise_dist(noise_generator);
                landmark_detections_.push_back({distance, bearing, landmark.id_});
            }

            for (auto const lm : landmark_detections_)
            {
                std::cout << "id: " << lm.id << " range: " << lm.range << " bearing: " << lm.bearing << std::endl;
            }
        }
    }
};

struct Odometry
{
    raylib::Vector2 estimated_position_;
    float           estimated_angle_;

    void update(const float &linear_vel_meas, const float &angular_vel_meas)
    {
        estimated_angle_ += kDt * angular_vel_meas;
        estimated_position_.x += linear_vel_meas * kDt * cos(estimated_angle_);
        estimated_position_.y += linear_vel_meas * kDt * sin(estimated_angle_);
    }
};

class Visualizer
{
  public:
    static constexpr int kScreenWidth{800};
    static constexpr int kScreenHeight{450};

    Visualizer()
    {
        window_ = std::make_unique<raylib::Window>(kScreenWidth, kScreenHeight, "2D Robot Localization Simulator");

        SetTargetFPS(60);
    };

    bool shouldClose()
    {
        return window_->ShouldClose();
    }

    void draw(const Agent &agent, const std::vector<Landmark> &landmarks, const Odometry &odometry)
    {
        BeginDrawing();
        ClearBackground(BLACK);
        // Draw the agent
        {
            DrawCircle(agent.position_.x, agent.position_.y, Agent::kRadius, BLUE); //Draw the robot
            DrawCircle(odometry.estimated_position_.x,
                       odometry.estimated_position_.y,
                       Agent::kRadius,
                       Fade(SKYBLUE, 0.3)); //  draw the odometry estimated
            DrawCircleLines(
                agent.position_.x, agent.position_.y, Agent::kSensorRange, GREEN); // Outline of sensing range
            DrawCircleV(raylib::Vector2(agent.position_.x, agent.position_.y),
                        Agent::kSensorRange,
                        Fade(LIGHTGRAY, 0.3)); // Sensing range visualization
            DrawLine(agent.position_.x,
                     agent.position_.y,
                     agent.position_.x + Agent::kSensorRange * cos(agent.angle_),
                     agent.position_.y + Agent::kSensorRange * sin(agent.angle_),
                     GREEN);
        }
        // Draw the landmarks
        {
            for (auto const &landmark : landmarks)
            {
                DrawCircle(landmark.position_.x, landmark.position_.y, 5, RED);
            }
        }

        EndDrawing();
    }

    std::unique_ptr<raylib::Window> window_;
};

int main()
{

    Agent                 agent{{400, 225}, 0.0};
    Odometry              odometry{agent.position_, agent.angle_};
    std::vector<Landmark> landmarks =
        generateLandmarks(10, Visualizer::kScreenWidth, Visualizer::kScreenHeight, Visualizer::kScreenHeight / 100);

    Visualizer visualizer;

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

        std::cout << "Robot: " << agent.position_.x << " " << agent.position_.y << std::endl;
        std::cout << "Odom : " << odometry.estimated_position_.x << " " << odometry.estimated_position_.y << std::endl;

        visualizer.draw(agent, landmarks, odometry);

        agent.senseLandmarks(landmarks);
    }

    std::cout << "done" << std::endl;

    return 0;
}
