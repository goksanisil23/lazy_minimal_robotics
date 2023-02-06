#pragma once

#include <algorithm>

#include "TimeUtil.h"
#include "raylib-cpp.hpp"

// 2D simulation environment of multiple robot that bounce from the walls of the environment.
// Robots mainly move with constant velocity profile with some random noise. The environment
// is rendered with raylib.
class MultiRobotEnv
{
  public:
    // --------- Constants --------- //
    static constexpr double ROBOT_RADIUS          = 4;
    static constexpr double ROBOT_BBOX_SIZE_RATIO = 2.5; // bbox_radius/robot_radius

    static constexpr double VX = 15;
    static constexpr double VY = 15;

    static constexpr int screenWidth  = 1200;
    static constexpr int screenHeight = 1000;

    static constexpr int FPS = 60;
    // ---------- Class Types -------------- //

    struct Robot
    {
        Robot(const raylib::Vector2 &pos0, const float &radius);

        void Draw() const;

        raylib::Vector2     position;
        raylib::BoundingBox bbox;
        float               radius;
    };

    // ---------- Member Variables -------------- //
    std::vector<Robot> robots_;

    // ---------- Member Functions -------------- //

    MultiRobotEnv();

    void GenerateRobots(const size_t &areaWidth, const size_t &areaHeight, const size_t &numRobots);
    void DrawRobots();
    void MoveRobots();
    void UpdateRobotBbox(Robot &robot);

    template <typename... Tfunc>
    bool StepEnv(Tfunc... userFuncs)
    {
        // -------- Draw -------- //
        BeginDrawing();

        simWindow_.ClearBackground(GRAY);

        (userFuncs(), ...);

        DrawFPS(10, 10);

        EndDrawing();

        return !(simWindow_.ShouldClose());
    }

  private:
    raylib::Window simWindow_;
};