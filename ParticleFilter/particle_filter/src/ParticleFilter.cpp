#include "ParticleFilter.h"

ParticleFilter::ParticleFilter(const std::string &mapFilePath)
{
    map_ = std::make_unique<landmarkSim2D::Map>(mapFilePath);
}
