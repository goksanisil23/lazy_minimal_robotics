#pragma once

#include <algorithm>

#include "Eigen/Core"
#include "TimeUtil.h"
#include "raylib-cpp.hpp"

// 2D simulation environment of multiple robot that bounce from the walls of the environment.
// Robots mainly move with constant velocity profile with some random noise. The environment
// is rendered with raylib.
class MultiRobotEnv
{
  public:
    // --------- Constants --------- //
    static constexpr double ROBOT_RADIUS          = 7;
    static constexpr double ROBOT_BBOX_SIZE_RATIO = 2.5; // bbox_radius/robot_radius

    static constexpr double V_MAX       = 10.0;
    static constexpr double SIM_STEP_DT = 0.01; // 100Hz

    static constexpr int SCREENWIDTH  = 1200;
    static constexpr int SCREENHEIGHT = 1000;

    static constexpr int FPS = 60;
    // ---------- Class Types -------------- //

    struct Robot
    {
        Robot(const raylib::Vector2 &pos0,
              const double          &heading0,
              const double          &v0,
              const double          &radius,
              const size_t          &id);

        void Draw() const;

        raylib::Vector2     position;
        double              heading;
        double              velocity;
        raylib::BoundingBox bbox;
        double              radius;
        size_t              id;
    };

    // ---------- Member Variables -------------- //
    std::vector<Robot> robots_;

    // ---------- Member Functions -------------- //

    MultiRobotEnv();

    void GenerateRobots(const size_t &areaWidth, const size_t &areaHeight, const size_t &numRobots);
    void DrawRobots();
    void MoveRobots(const double &dt);
    void UpdateRobotBbox(Robot &robot);

    std::vector<Eigen::VectorXd> GetAllRobotBboxs();

    template <typename... Tfunc>
    bool RenderEnv(Tfunc... userFuncs)
    {
        // -------- Draw -------- //
        BeginDrawing();

        simWindow_.ClearBackground(GRAY);

        (userFuncs(), ...);

        DrawFPS(10, 10);

        EndDrawing();

        return !(simWindow_.ShouldClose());
    }

  public:
    double areaWidth_, areaHeight_;

  private:
    raylib::Window simWindow_;
};