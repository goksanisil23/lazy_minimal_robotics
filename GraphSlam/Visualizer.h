#pragma once

#include "raylib-cpp.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "Agent.h"
#include "Landmark.h"
#include "Odometry.h"

class Visualizer
{
  public:
    static constexpr int kScreenWidth{800};
    static constexpr int kScreenHeight{450};

    Visualizer();

    bool shouldClose();

    void draw(const Agent                             &agent,
              const std::vector<Landmark>             &landmarks,
              raylib::Vector2                         &dead_reckon,
              raylib::Vector2                         &opt_pose,
              std::unordered_map<int, Eigen::Vector2d> landmarks_slam);

    std::unique_ptr<raylib::Window> window_;
};