#pragma once

#include <vector>

#include "Eigen/Dense"
#include "open3d/Open3D.h"

#include <carla/client/Sensor.h>
#include <carla/sensor/data/LidarData.h>

namespace csd           = carla::sensor::data;
using SemanticLidarData = csd::Array<csd::SemanticLidarDetection>;

class OccupancyGrid
{

  public:
    OccupancyGrid(const int &numCellsX, const int &numCellsY, const float &resolution);
    void UpdateGrid(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr);
    void InitVisualizer();
    void ShowGrid();

  private:
    const int   NUM_CELLS_X_;
    const int   NUM_CELLS_Y_;
    const float GRID_RESOLUTION_; // [m.]

    std::vector<std::vector<float>> grid_;

    open3d::visualization::Visualizer gridViz_;
};