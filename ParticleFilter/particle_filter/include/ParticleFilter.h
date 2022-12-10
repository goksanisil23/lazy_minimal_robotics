#pragma once

#include <memory>
#include <string>

#include <landmarkSim2dLib/Map.h>

class ParticleFilter
{
  public:
    ParticleFilter(const std::string &mapFilePath);

  private:
    std::unique_ptr<landmarkSim2D::Map> map_;
};