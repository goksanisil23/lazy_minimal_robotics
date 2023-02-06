#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>

#include "raylib-cpp.hpp"

#include "HungarianOptimizer.hpp"
#include "MultiRobotEnv.h"

constexpr size_t NUM_ROBOTS = 10;

// row: robot, col: hole
// std::vector<double> GenerateCostVector(const std::vector<MultiRobotEnv::Robot> &robots,
//                                        std::vector<MultiRobotEnv::Hole>        &holes)
// {
//     std::vector<double> costVec;
//     for (const auto &robot : robots)
//     {
//         for (const auto &hole : holes)
//         {
//             costVec.push_back(MultiRobotEnv::GetRobotHoleDistance2(robot, hole));
//         }
//     }

//     return costVec;
// }

// first: row, second: col
void ShowAssignments(const std::vector<std::pair<size_t, size_t>> &assignments)
{
    for (const auto &assignment : assignments)
    {
        std::cout << assignment.first << " -> " << assignment.second << std::endl;
    }
}

int main(void)
{
    MultiRobotEnv env;
    env.GenerateRobots(GetScreenWidth(), GetScreenHeight(), NUM_ROBOTS);

    // Create Hungarian solver for assignment problem
    std::vector<std::pair<size_t, size_t>> robotToHoleMatches;
    HungarianOptimizer<double>             hungarianSolver;
    hungarianSolver.costs()->Reserve(1000, 1000);
    // hungarianSolver.costs()->Resize(robots.size(), holes.size());
    // std::vector<double> costVec(GenerateCostVector(robots, holes));
    // hungarianSolver.costs()->AssignFromVec(costVec);
    // hungarianSolver.Minimize(&robotToHoleMatches);
    // ShowAssignments(robotToHoleMatches);

    // auto drawFunc = std::bind(DrawMatches, std::ref(robotToHoleMatches), std::ref(robots));
    auto moveFunc = std::bind(&MultiRobotEnv::MoveRobots, &env);
    auto drawFunc = std::bind(&MultiRobotEnv::DrawRobots, &env);
    // auto moveFunc = std::bind(MoveRobots, &robots, 1.0); // alternatively use pointer

    bool simOk = true;
    while (simOk)
    {

        simOk = env.StepEnv(moveFunc, drawFunc);
        std::cout << env.robots_.at(0).position.x << " " << env.robots_.at(0).position.y << std::endl;
        // std::cout << GetFrameTime() << std::endl;
        // simOk = env.StepEnv(moveRobots(0.1));
    }

    return 0;
}
