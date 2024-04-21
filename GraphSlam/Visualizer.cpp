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
                      DeadReckon                              &dead_reckon,
                      Eigen::Vector3d                         &opt_pose,
                      std::unordered_map<int, Eigen::Vector2d> landmarks_slam)
{
    const auto dead_reckon_color  = Fade(MAGENTA, 0.5);
    const auto actual_agent_color = BLUE;
    const auto graph_slam_color   = Fade(YELLOW, 0.7);

    BeginDrawing();
    ClearBackground(BLACK);
    // Drawing agent and estimators
    {
        DrawCircle(dead_reckon.position.x, dead_reckon.position.y, Agent::kRadius, dead_reckon_color);
        DrawLine(dead_reckon.position.x,
                 dead_reckon.position.y,
                 dead_reckon.position.x + Agent::kRadius * cos(dead_reckon.heading),
                 dead_reckon.position.y + Agent::kRadius * sin(dead_reckon.heading),
                 WHITE);

        DrawCircle(opt_pose(0), opt_pose(1), Agent::kRadius, Fade(YELLOW, 0.7));
        DrawLine(opt_pose(0),
                 opt_pose(1),
                 opt_pose(0) + Agent::kRadius * cos(opt_pose(2)),
                 opt_pose(1) + Agent::kRadius * sin(opt_pose(2)),
                 BLACK);

        DrawCircle(agent.position_.x, agent.position_.y, Agent::kRadius * 0.7, BLUE);      //Draw the robot
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
    // Legendbox
    {
        constexpr float   kBoxHeight{80};
        constexpr float   kBoxWidth{100};
        raylib::Rectangle legendBox(0, kScreenHeight - kBoxHeight, kBoxWidth, kBoxHeight);
        legendBox.Draw(Fade(GRAY, 0.3));

        auto drawLabel = [](const std::string &label, const int i, const raylib::Color color)
        {
            int circle_y = kScreenHeight - (i * 20);
            DrawCircle(15, circle_y, 10, color);
            DrawText(label.c_str(), 30, circle_y - 10, 10, color);
        };

        drawLabel("actual", 1, actual_agent_color);
        drawLabel("dead reckon", 2, dead_reckon_color);
        drawLabel("graph slam", 3, graph_slam_color);
    }

    EndDrawing();
}
