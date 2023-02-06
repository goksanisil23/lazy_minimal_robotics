#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>

#include "raylib-cpp.hpp"

#include "Environment.hpp"
#include "HungarianOptimizer.hpp"
#include "TimeUtil.h"

constexpr double VX = 15;
constexpr double VY = 15;

// row: robot, col: hole
std::vector<double> GenerateCostVector(const std::vector<Env::Robot> &robots, std::vector<Env::Hole> &holes)
{
    std::vector<double> costVec;
    for (const auto &robot : robots)
    {
        for (const auto &hole : holes)
        {
            costVec.push_back(Env::GetRobotHoleDistance2(robot, hole));
        }
    }

    return costVec;
}

// first: row, second: col
void ShowAssignments(const std::vector<std::pair<size_t, size_t>> &assignments)
{
    for (const auto &assignment : assignments)
    {
        std::cout << assignment.first << " -> " << assignment.second << std::endl;
    }
}

void MoveRobots(std::vector<Env::Robot> &robots)
{
    static auto prevTime = time_util::chronoNow();
    auto        currTime = time_util::chronoNow();
    double      dt       = time_util::getTimeDuration(currTime, prevTime);

    std::for_each(
        robots.begin(), robots.end(), [&dt](Env::Robot &robot) { robot.position += raylib::Vector2(VX, VY) * dt; });

    prevTime = currTime;
}

void DrawMatches(const std::vector<std::pair<size_t, size_t>> &robotToHoleMatches,
                 const std::vector<Env::Robot>                &robots,
                 const std::vector<Env::Hole>                 &holes)
{
    std::for_each(holes.begin(), holes.end(), [](const Env::Hole &hole) { hole.Draw(); });
    std::for_each(robots.begin(), robots.end(), [](const Env::Robot &robot) { robot.Draw(); });
    std::for_each(robotToHoleMatches.begin(),
                  robotToHoleMatches.end(),
                  [&holes, &robots](const auto &robotHolePair)
                  {
                      const Env::Robot     &robot = robots.at(robotHolePair.first);
                      const Env::Hole      &hole  = holes.at(robotHolePair.second);
                      const raylib::Vector3 startPt{robot.position.x, robot.position.y, 0};
                      const raylib::Vector3 endPt{hole.position.x, hole.position.y};
                      DrawLine(startPt.x, startPt.y, endPt.x, endPt.y, GREEN);
                  });
}

int main(void)
{
    Env env;

    std::vector<Env::Hole>  holes{env.GenerateHoles(GetScreenWidth(), GetScreenHeight(), 10)};
    std::vector<Env::Robot> robots{env.GenerateRobots(GetScreenWidth(), GetScreenHeight(), 10)};

    // Create Hungarian solver for assignment problem
    std::vector<std::pair<size_t, size_t>> robotToHoleMatches;
    HungarianOptimizer<double>             hungarianSolver;
    hungarianSolver.costs()->Reserve(1000, 1000);
    hungarianSolver.costs()->Resize(robots.size(), holes.size());
    std::vector<double> costVec(GenerateCostVector(robots, holes));
    hungarianSolver.costs()->AssignFromVec(costVec);
    hungarianSolver.Minimize(&robotToHoleMatches);
    ShowAssignments(robotToHoleMatches);

    auto drawFunc = std::bind(DrawMatches, std::ref(robotToHoleMatches), std::ref(robots), std::ref(holes));
    auto moveFunc = std::bind(MoveRobots, std::ref(robots));
    // auto moveFunc = std::bind(MoveRobots, &robots, 1.0); // alternatively use pointer

    bool simOk = true;
    while (simOk)
    {

        simOk = env.StepEnv(drawFunc, moveFunc);
        // simOk = env.StepEnv(drawFunc, moveFunc2);
        std::cout << robots.at(0).position.x << " " << robots.at(0).position.y << std::endl;
        // std::cout << GetFrameTime() << std::endl;
        // simOk = env.StepEnv(moveRobots(0.1));
    }

    return 0;
}
