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
    OccupancyGrid(const int   &numRows,
                  const int   &numCols,
                  const float &resolution,
                  const float &alpha,
                  const float &beta);
    void UpdateGrid(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr);
    void InitGrid();
    void InitVisualizer();
    void UpdateCellVis(const int &col, const int &row, const Eigen::Vector3d &color);
    void InverseMeasModel();
    void UpdateGridVis();
    void ShowGrid();

    std::vector<std::pair<int, int>> BresenhamLineCells(const int &x0, const int &y0, const int &x1, const int &y1);

    size_t FindClosestBearingRayToCell(const float &cellBearing, const std::vector<float> &sensorRayBearings);

    void CalculateSensorRayRangeAndBearings(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr,
                                            std::vector<float>                         &sensorRayBearings,
                                            std::vector<float>                         &sensorRayRanges);

  private:
    const int   NUM_ROWS_;        // forward
    const int   NUM_COLS_;        // left
    const float GRID_RESOLUTION_; // [m.]

    const float ALPHA_; // [m.] determines the effected range for the cone
    const float BETA_;  // [rad.] determines the effected angle for the cone

    const float SENSOR_RANGE_; // [m.]

    struct Cell
    {
        Cell() = default;
        // Cell(const float &y, const float &x) : y_{y}, x_{x}
        // {
        // }

        // Cell operator+(const Cell &other) const
        // {
        //     return Cell{this->y_ + other.y_, this->x_ + other.x_};
        // }

        float probOcc; // occupancy probability of the cell
        // int   rowIdx, colIdx;   // row/col idx of the cell within the grid
        float centerX, centerY; // center coordinate of the cell
        float range;            // distance of the cell w.r.t sensor
        float bearing;          // beaering of the cell w.r.t the sensor
    };

    // Grid coordinate system: X-forward, y-left, z-up. Bottom-middle edge is where sensor is.
    // (N_COLS-1,N_ROWS-1)-----------------
    // ------------------------------------
    // ------------------------------------
    // ----------------v---------------(0,0)

    std::vector<std::vector<Cell>> grid_;

    open3d::visualization::Visualizer               gridViz_;
    std::shared_ptr<open3d::geometry::TriangleMesh> triangleMesh_;
};