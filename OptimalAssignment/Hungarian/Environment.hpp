#pragma once

#include "raylib-cpp.hpp"

class Env
{
  public:
    // --------- Constants --------- //
    static constexpr float HOLE_SIZE    = 10;
    static constexpr float ROBOT_RADIUS = 4;

    static constexpr int screenWidth  = 1200;
    static constexpr int screenHeight = 1000;

    static constexpr int FPS = 60;
    // ----------------------------- //

    struct Hole
    {
        Hole(const raylib::Rectangle &bbox) : bbox{bbox}
        {
            position.x = bbox.x + bbox.width / 2.0f;
            position.y = bbox.y + bbox.height / 2.0f;
        }

        void Draw() const
        {
            bbox.Draw(GOLD);
        }

        raylib::Rectangle bbox;
        raylib::Vector2   position;
        bool              occupied{false};
    };

    struct Robot
    {
        Robot(const raylib::Vector2 &pos0, const float &radius) : position{pos0}, radius{radius}
        {
        }

        void Draw() const
        {
            DrawCircle(position.x, position.y, ROBOT_RADIUS, BLUE);
        }

        raylib::Vector2 position;
        float           radius;
        bool            reached{false};
    };

    // ------------------------------ //

    Env() : simWindow(screenWidth, screenHeight, "Hungarian Assignment")
    {
        SetTargetFPS(FPS);
    }

    // Generates random holes, within the render window
    std::vector<Hole> GenerateHoles(const size_t &areaWidth, const size_t &areaHeight, const size_t &numHoles = 10)
    {
        std::vector<Hole> holes;
        for (size_t holeIdx = 0; holeIdx < numHoles; holeIdx++)
        {
            float xCoord = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (areaWidth));
            float yCoord = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (areaHeight));
            holes.emplace_back(
                Hole(raylib::Rectangle(raylib::Vector2(xCoord, yCoord), raylib::Vector2(HOLE_SIZE, HOLE_SIZE))));
        }

        return holes;
    }

    std::vector<Robot> GenerateRobots(const size_t &areaWidth, const size_t &areaHeight, const size_t &numRobots = 10)
    {
        std::vector<Robot> robots;
        for (size_t robotIdx = 0; robotIdx < numRobots; robotIdx++)
        {
            float xCoord = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (areaWidth));
            float yCoord = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (areaHeight));
            robots.emplace_back(Robot(raylib::Vector2(xCoord, yCoord), ROBOT_RADIUS));
        }

        return robots;
    }

    static double GetRobotHoleDistance2(const Robot &robot, const Hole &hole)
    {
        return (robot.position.x - hole.position.x) * (robot.position.x - hole.position.x) +
               (robot.position.y - hole.position.y) * (robot.position.y - hole.position.y);
    }

    template <typename... Tfunc>
    bool StepEnv(Tfunc... userFuncs)
    {
        // -------- Draw -------- //
        BeginDrawing();

        simWindow.ClearBackground(GRAY);

        (userFuncs(), ...);

        DrawFPS(10, 10);

        EndDrawing();

        return !(simWindow.ShouldClose());
    }

  private:
    raylib::Window simWindow;
};