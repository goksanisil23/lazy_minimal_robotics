#include "Visualizer.h"

Visualizer::Visualizer()
{
    window_ = std::make_unique<raylib::Window>(kScreenWidth, kScreenHeight, "2D Robot Localization Simulator");

    SetTargetFPS(60);
};

bool Visualizer::shouldClose()
{
    return window_->ShouldClose();
}

void Visualizer::draw(const Agent                             &agent,
                      const std::vector<Landmark>             &landmarks,
                      raylib::Vector2                         &dead_reckon,
                      raylib::Vector2                         &opt_pose,
                      std::unordered_map<int, Eigen::Vector2d> landmarks_slam)
{
    BeginDrawing();
    ClearBackground(BLACK);
    // Draw the agent
    {
        DrawCircle(agent.position_.x, agent.position_.y, Agent::kRadius, BLUE);       //Draw the robot
        DrawCircle(dead_reckon.x, dead_reckon.y, Agent::kRadius, Fade(SKYBLUE, 0.3)); //  draw the odometry estimated
        DrawCircle(opt_pose.x, opt_pose.y, Agent::kRadius, Fade(YELLOW, 0.7));        //  draw the odometry estimated
        DrawCircleLines(agent.position_.x, agent.position_.y, Agent::kSensorRange, GREEN); // Outline of sensing range
        DrawCircleV(raylib::Vector2(agent.position_.x, agent.position_.y),
                    Agent::kSensorRange,
                    Fade(LIGHTGRAY, 0.3)); // Sensing range visualization
        DrawLine(agent.position_.x,
                 agent.position_.y,
                 agent.position_.x + Agent::kSensorRange * cos(agent.heading_),
                 agent.position_.y + Agent::kSensorRange * sin(agent.heading_),
                 GREEN);
    }
    // Draw the landmarks
    {
        for (auto const &landmark : landmarks)
        {
            DrawCircle(landmark.position_.x, landmark.position_.y, Landmark::kRadius, RED);
        }
    }
    // Draw the estimated landmarks from slam
    {
        for (auto const &lm : landmarks_slam)
        {
            DrawCircle(lm.second(0), lm.second(1), Landmark::kRadius * 2.0, Fade(ORANGE, 0.3));
        }
    }

    EndDrawing();
}
