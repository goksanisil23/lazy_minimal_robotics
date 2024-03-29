#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>

#include "argparse.hpp"
#include "raylib-cpp.hpp"

#include "MultiRobotEnv.h"
#include "SortTracker/SortTracker.h"

int main(int argc, char *argv[])
{

    argparse::ArgumentParser program("sort_tracker");
    program.add_argument("--num_robots").required().help("number of robots to be tracked").scan<'u', size_t>();
    program.parse_args(argc, argv);
    int numRobots = program.get<size_t>("--num_robots");

    MultiRobotEnv env;
    env.GenerateRobots(env.areaWidth_, env.areaHeight_, numRobots);

    auto drawEnvFunc = std::bind(&MultiRobotEnv::DrawRobots, &env);

    // Sort Tracker
    SortTracker sortTracker;

    bool simOk = true;
    int  ctr   = 0;
    while (simOk)
    {
        env.MoveRobots(env.SIM_STEP_DT);
        auto bboxDetections = env.GetAllRobotBboxs();
        if ((ctr > 10) && (ctr < 30))
            bboxDetections = std::vector<Eigen::VectorXd>(bboxDetections.begin() + 1, bboxDetections.end());
        sortTracker.Step(bboxDetections, env.SIM_STEP_DT);

        simOk = env.RenderEnv(drawEnvFunc);
        ctr++;
        std::cout << "ctr: " << ctr << std::endl;
    }

    return 0;
}
