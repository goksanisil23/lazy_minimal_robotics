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
    void UpdateGridNaive(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr);
    void UpdateGridBresenham(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr);
    void InitGrid();
    void InitGridVisualizer();
    void UpdateCellVis(const int &col, const int &row, const Eigen::Vector3d &color);
    void InverseMeasModel();
    void UpdateCloudViz(boost::shared_ptr<SemanticLidarData> pointcloudPtr);
    void UpdateGridVis();
    void ShowGrid();

    inline bool IsGroundHit(const carla::sensor::data::SemanticLidarDetection &hit);
    inline bool IsAboveSensor(const carla::sensor::data::SemanticLidarDetection &hit);

    std::tuple<int, int, bool> DiscretizePointToCell(const float &x, const float &y);

    std::vector<std::pair<int, int>> BresenhamLineCells(const int &x0, const int &y0, const int &x1, const int &y1);

    size_t FindClosestBearingRayToCell(const float &cellBearing, const std::vector<float> &sensorRayBearings);

    void CalculateSensorRayRangeAndBearings(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr,
                                            std::vector<float>                         &sensorRayBearings,
                                            std::vector<float>                         &sensorRayRanges);

    void SortAndDiscretizePointloud(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr,
                                    std::vector<std::tuple<int, int, bool>>    &hitCells,
                                    std::multimap<float, size_t>               &rangeSortedCloudIndices);

  private:
    const int   NUM_ROWS_;        // forward
    const int   NUM_COLS_;        // left
    const float GRID_RESOLUTION_; // [m.]

    const float ALPHA_; // [m.] determines the effected range for the cone
    const float BETA_;  // [rad.] determines the effected angle for the cone

    // sensor position in grid coordinate system.
    // Note that cell(0,0) does NOT correspond to x=0,y=0, since we center the grid at the sensor
    // But the cell indices still start from bottom right
    const float SENSOR_RANGE_;   // [m.]
    const float SENSOR_POS_X_;   // [m.]
    const float SENSOR_POS_Y_;   // [m.]
    int         SENSOR_POS_COL_; // sensor position within the grid
    int         SENSOR_POS_ROW_; // sensor position within the grid

    struct Grid
    {
        Grid() = default;

        // 2D grid data is stored row-wise (fill columns in a row, move to next row)
        std::vector<float> probOcc;          // occupancy probability of the cells
        std::vector<float> centerX, centerY; // center coordinate of the cells
        std::vector<float> range;            // distance of the cells w.r.t sensor
        std::vector<float> bearing;          // beaering of the cells w.r.t the sensor
    };

    // Grid coordinate system: X-forward, y-left, z-up. Bottom-middle edge is where sensor is.
    // (N_COLS-1,N_ROWS-1)-----------------
    // ------------------------------------
    // ------------------------------------
    // ----------------v---------------(0,0)

    Grid grid_;

    open3d::visualization::Visualizer               gridViz_;
    std::shared_ptr<open3d::geometry::TriangleMesh> triangleMesh_;
};