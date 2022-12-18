#pragma once

#include <memory>

#include "Map.h"
#include "Robot.h"

namespace landmarkSim2D
{

const std::string MAP_PATH = "/tmp/landmark_map_2d.txt";

class Sim
{
  public:
    Sim();

    void Init();
    void Step(const float &dt);

    std::shared_ptr<Map>   map;
    std::shared_ptr<Robot> robot;

  private:
};

} // namespace landmarkSim2D