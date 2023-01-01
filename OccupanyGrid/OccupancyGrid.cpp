#include "OccupancyGrid.h"

#include <algorithm>
#include <functional>
#include <map>
#include <math.h>
#include <random>

#include "TimeUtil.h"

OccupancyGrid::OccupancyGrid(const int   &numRows,
                             const int   &numCols,
                             const float &resolution,
                             const float &alpha,
                             const float &beta)
    : NUM_ROWS_{numRows}, NUM_COLS_{numCols}, GRID_RESOLUTION_{resolution}, ALPHA_{alpha}, BETA_{beta},
      SENSOR_RANGE_{100.0f}, SENSOR_POS_X_{0.0f}, SENSOR_POS_Y_{0.0f}
{
    assert(NUM_COLS_ % 2 == 0);

    InitGrid();
    InitGridVisualizer();
}

void OccupancyGrid::InitGrid()
{

    grid_.probOcc.resize(NUM_COLS_ * NUM_ROWS_, 0.5);
    grid_.range.resize(NUM_COLS_ * NUM_ROWS_);
    grid_.bearing.resize(NUM_COLS_ * NUM_ROWS_);
    grid_.centerX.resize(NUM_COLS_ * NUM_ROWS_);
    grid_.centerY.resize(NUM_COLS_ * NUM_ROWS_);

    // // X-forward, y-left. Grid(0,0) at bottom right
    static float GRID_ORIGO_Y = -(float)(NUM_COLS_ / 2) * GRID_RESOLUTION_;
    static float GRID_ORIGO_X = 0.0f;

    std::tuple<int, int, bool> sensorGridPos{DiscretizePointToCell(SENSOR_POS_X_, SENSOR_POS_Y_)};
    assert(std::get<0>(sensorGridPos)); // sensor must be within the grid
    SENSOR_POS_ROW_ = std::get<0>(sensorGridPos);
    SENSOR_POS_COL_ = std::get<1>(sensorGridPos);

    for (int row = 0; row < NUM_ROWS_; row++)
    {
        for (int col = 0; col < NUM_COLS_; col++)
        {
            size_t cellIdx            = row * NUM_COLS_ + col;
            grid_.centerX.at(cellIdx) = (float)row * GRID_RESOLUTION_ + GRID_ORIGO_X;
            grid_.centerY.at(cellIdx) = (float)col * GRID_RESOLUTION_ + GRID_ORIGO_Y;
            grid_.range.at(cellIdx)   = std::sqrt(std::pow(grid_.centerY.at(cellIdx) - SENSOR_POS_Y_, 2) +
                                                std::pow(grid_.centerX.at(cellIdx) - SENSOR_POS_X_, 2));
            grid_.bearing.at(cellIdx) = std::fmod((std::atan2(grid_.centerY.at(cellIdx) - SENSOR_POS_Y_,
                                                              grid_.centerX.at(cellIdx) - SENSOR_POS_X_)) +
                                                      M_PI,
                                                  2.0 * M_PI) -
                                        M_PI;
        }
    }
}

void OccupancyGrid::InitGridVisualizer()
{
    gridViz_.CreateVisualizerWindow("Grid", 720, 1280, 1700, 270);
    gridViz_.GetRenderOption().background_color_      = {0.05, 0.05, 0.05};
    gridViz_.GetRenderOption().point_size_            = 1;
    gridViz_.GetRenderOption().show_coordinate_frame_ = true;

    // Eigen::Vector3i voxelIdx(col, row, 0);

    const double GRID_VIS_RESOLUTION{((double)GRID_RESOLUTION_)};

    triangleMesh_ = std::make_shared<open3d::geometry::TriangleMesh>();
    for (int row = 0; row < NUM_ROWS_; row++)
    {
        for (int col = 0; col < NUM_COLS_; col++)
        {
            // Create a quad for each grid cell
            std::vector<Eigen::Vector3d> vertices;
            vertices.emplace_back((double)((NUM_COLS_ - 1) - col) * (double)GRID_VIS_RESOLUTION,
                                  (double)row * (double)GRID_VIS_RESOLUTION,
                                  0.0);
            vertices.emplace_back((double)((NUM_COLS_ - 1) - col + 1) * (double)GRID_VIS_RESOLUTION,
                                  (double)row * (double)GRID_VIS_RESOLUTION,
                                  0.0);
            vertices.emplace_back((double)((NUM_COLS_ - 1) - col + 1) * (double)GRID_VIS_RESOLUTION,
                                  (double)(row + 1) * (double)GRID_VIS_RESOLUTION,
                                  0.0);
            vertices.emplace_back((double)((NUM_COLS_ - 1) - col) * (double)GRID_VIS_RESOLUTION,
                                  (double)(row + 1) * (double)GRID_VIS_RESOLUTION,
                                  0.0);

            std::vector<Eigen::Vector3i> triangles;
            triangles.emplace_back((int)triangleMesh_->vertices_.size(),
                                   (int)triangleMesh_->vertices_.size() + 1,
                                   (int)triangleMesh_->vertices_.size() + 2);
            triangles.emplace_back((int)triangleMesh_->vertices_.size() + 2,
                                   (int)triangleMesh_->vertices_.size() + 3,
                                   (int)triangleMesh_->vertices_.size());

            triangleMesh_->vertices_.insert(triangleMesh_->vertices_.end(), vertices.begin(), vertices.end());
            triangleMesh_->triangles_.insert(triangleMesh_->triangles_.end(), triangles.begin(), triangles.end());

            // Set the color to gray
            for (size_t k = 0; k < 4; k++)
            {
                triangleMesh_->vertex_colors_.emplace_back(0.5, 0.5, 0.5);
            }
        }
    }
    gridViz_.AddGeometry(triangleMesh_);
}

size_t OccupancyGrid::FindClosestBearingRayToCell(const float &cellBearing, const std::vector<float> &sensorRayBearings)
{
    // Need to iterate all points in the point cloud
    // 1) Calculate bearing of each point (ray)
    // 2) Check if its closer to the cell bearing
    float  minDiff{2 * M_PI};
    float  sensorRayBearing;
    float  bearingDiff;
    size_t ptIdx = 0;
    size_t minBearingPtIdx;
    for (const auto &sensorRayBearing : sensorRayBearings)
    {
        bearingDiff = std::abs(sensorRayBearing - cellBearing);
        if (bearingDiff < minDiff)
        {
            minDiff         = bearingDiff;
            minBearingPtIdx = ptIdx;
        }
        ptIdx++;
    }
    return minBearingPtIdx;
}

void OccupancyGrid::CalculateSensorRayRangeAndBearings(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr,
                                                       std::vector<float>                         &sensorRayBearings,
                                                       std::vector<float>                         &sensorRayRanges)
{
    size_t ptIdx = 0;
    for (const auto &pt : *pointcloudPtr)
    {
        sensorRayRanges.at(ptIdx)   = std::sqrt(pt.point.y * pt.point.y + pt.point.x * pt.point.x);
        sensorRayBearings.at(ptIdx) = std::fmod((std::atan2(pt.point.y, pt.point.x)) + M_PI, 2.0 * M_PI) - M_PI;
        ptIdx++;
    }
}

void OccupancyGrid::UpdateGridNaive(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr)
{
    // TODO: Ignore points higher than vehicle's height.

    std::vector<float> sensorRayBearings(pointcloudPtr->size());
    std::vector<float> sensorRayRanges(pointcloudPtr->size());

    auto t0 = time_util::chronoNow();
    CalculateSensorRayRangeAndBearings(pointcloudPtr, sensorRayBearings, sensorRayRanges);
    auto t1 = time_util::chronoNow();

    for (int row = 0; row < NUM_ROWS_; row++)
    {
        for (int col = 0; col < NUM_COLS_; col++)
        {
            size_t cellIdx = row * NUM_COLS_ + col;
            // Find the sensor measurement (ray) that is closest in angle to this cell
            size_t k{FindClosestBearingRayToCell(grid_.bearing.at(cellIdx), sensorRayBearings)};

            // Check:
            // If cell distance greater than maximum sensor range
            // If cell distance is behind the associated sensor ray + ALPHA_region
            // If cell is outside the field of view of the associated sensor ray + BETA_REGION
            if ((grid_.range.at(cellIdx) > std::fmin(SENSOR_RANGE_, sensorRayRanges.at(k) + ALPHA_ / 2.0f)) ||
                ((std::fabs(grid_.bearing.at(cellIdx) - sensorRayBearings.at(k)) > (BETA_ / 2.0f))))
            {
                grid_.probOcc.at(cellIdx) = 0.5f;
            }
            // If sensor ray measurement lies within this cell (+ALPHA region) = OCCUPIED
            else if ((sensorRayRanges.at(k) < SENSOR_RANGE_) &&
                     (std::fabs(sensorRayRanges.at(k) - grid_.range.at(cellIdx)) < (ALPHA_ / 2.0f)))
            {
                grid_.probOcc.at(cellIdx) = 0.75f;
            }
            // If the sensor ray measurement is behind the cell = UNOCCUPIED
            else if (sensorRayRanges.at(k) > grid_.range.at(cellIdx))
            {
                grid_.probOcc.at(cellIdx) = 0.25f;
            }
            else
            {
                std::cerr << "should not come here !!\n";
            }
        }
    }

    auto t2 = time_util::chronoNow();

    time_util::showTimeDuration(t1, t0, "sensor ray/range: ");
    time_util::showTimeDuration(t2, t1, "occupancy loop  : ");
}

// Discretize the point coordinates in space to grid cell indices, check if point lies within the grid
// Returns (row,col,isWithinGrid)
std::tuple<int, int, bool> OccupancyGrid::DiscretizePointToCell(const float &x, const float &y)
{
    int gridColIdx = std::floor(y / GRID_RESOLUTION_ + NUM_COLS_ / 2);
    int gridRowIdx = std::floor(x / GRID_RESOLUTION_);
    // // Check if the point is within the grid
    // if ((gridRowIdx >= 0) && (gridRowIdx < NUM_ROWS_) && (gridColIdx >= 0) && (gridColIdx < NUM_COLS_))
    // {
    //     return std::tuple<int, int, bool>(gridRowIdx, gridColIdx, true);
    // }
    // Check if the point is within the grid
    if ((gridRowIdx >= 0) && (gridRowIdx < NUM_ROWS_) && (gridColIdx >= 0) && (gridColIdx < NUM_COLS_))
    {
        return std::tuple<int, int, bool>(gridRowIdx, gridColIdx, true);
    }
    else
    {
        return std::tuple<int, int, bool>(gridRowIdx, gridColIdx, false);
    }
}

inline bool OccupancyGrid::IsGroundHit(const carla::sensor::data::SemanticLidarDetection &hit)
{
    if ((hit.object_tag == static_cast<uint32_t>(carla::rpc::CityObjectLabel::Roads)) ||
        (hit.object_tag == static_cast<uint32_t>(carla::rpc::CityObjectLabel::RoadLines)))
    {
        return true;
    }
    return false;
}

inline bool OccupancyGrid::IsAboveSensor(const carla::sensor::data::SemanticLidarDetection &hit)
{
    return hit.point.z > 0.0;
}

void OccupancyGrid::SortAndDiscretizePointloud(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr,
                                               std::vector<std::tuple<int, int, bool>>    &hitCells,
                                               std::multimap<float, size_t>               &rangeSortedCloudIndices)
{
    size_t idx = 0;
    for (const auto &hit : *pointcloudPtr)
    {
        hitCells.at(idx) = std::move(DiscretizePointToCell(hit.point.x, hit.point.y));
        float hitRange2{hit.point.x * hit.point.x + hit.point.y * hit.point.y};
        rangeSortedCloudIndices.emplace(hitRange2, idx);
        idx++;
    }
}

void OccupancyGrid::UpdateGridBresenham(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr)
{
    std::vector<float> sensorRayBearings(pointcloudPtr->size());
    std::vector<float> sensorRayRanges(pointcloudPtr->size());

    // Reset the grid status
    auto t0 = time_util::chronoNow();
    std::fill(grid_.probOcc.begin(), grid_.probOcc.end(), 0.5);
    auto t1 = time_util::chronoNow();

    // Sort the pointcloud based on range
    std::vector<std::tuple<int, int, bool>> hitCells(pointcloudPtr->size());
    std::multimap<float, size_t>            rangeSortedCloudIndices;
    SortAndDiscretizePointloud(pointcloudPtr, hitCells, rangeSortedCloudIndices);
    auto t2 = time_util::chronoNow();

    // For each lidar hit, we trace the ray from the sensor to the hit.
    // Along the ray, we update cells as free until the hit cell.
    // If we see a cell with high occupancy along the ray already, we stop the traversal (leaving rest of the cells at 0.5)
    // NOTE: this only works if we loop from low range to high range hits
    for (const auto &el : rangeSortedCloudIndices)
    {
        size_t rangeSortedCloudIdx{el.second};

        const auto                       &hit{pointcloudPtr->at(rangeSortedCloudIdx)};
        const std::tuple<int, int, bool> &hitCell{hitCells.at(rangeSortedCloudIdx)};

        // If the cell coordinate is within bounding box AND below the sensor AND not ground
        if (std::get<2>(hitCell) && (hit.point.z <= 0.0) && !IsGroundHit(hit))
        {
            auto cellsAlongRay{
                BresenhamLineCells(SENSOR_POS_ROW_, SENSOR_POS_COL_, std::get<0>(hitCell), std::get<1>(hitCell))};

            for (int i = 0; i < cellsAlongRay.size() - 1; i++) // not including the hit point cell
            {
                int cellIdx = cellsAlongRay.at(i).first * NUM_COLS_ + cellsAlongRay.at(i).second;
                if ((grid_.probOcc.at(cellIdx) > 0.7))
                {
                    break;
                    // continue;
                }
                else
                {
                    grid_.probOcc.at(cellIdx) = 0.25;
                }
            }
            // For the last cell on the ray line, assign occupied since thats the hit cell
            int hitPtCellIdx = cellsAlongRay.back().first * NUM_COLS_ + cellsAlongRay.back().second;

            {
                grid_.probOcc.at(hitPtCellIdx) = 0.75;
            }
        }
    }
    auto t3 = time_util::chronoNow();

    time_util::showTimeDuration(t1, t0, "reset grid: ");
    time_util::showTimeDuration(t2, t1, "sort/disc:  ");
    time_util::showTimeDuration(t3, t2, "occloop  : ");
}

// void OccupancyGrid::InverseMeasModel()
// {
//     for (int row = 0; row < NUM_ROWS_; row++)
//     {
//         for (int col = 0; col < NUM_COLS_; col++)
//         {
//             float cellRange = std::sqrt(cell)
//         }
//     }
// }

// Given a starting cell (x0,y0) and ending cell(y1,y1), returns the cells inbetween that are on a connecting line
// Result contains both (x0,y0) and (x1,y1)
// https://github.com/encukou/bresenham/blob/master/bresenham.py
std::vector<std::pair<int, int>>
OccupancyGrid::BresenhamLineCells(const int &x0, const int &y0, const int &x1, const int &y1)
{
    int dx = x1 - x0;
    int dy = y1 - y0;

    int xsign = (dx > 0) ? 1 : -1;
    int ysign = (dy > 0) ? 1 : -1;

    dx = std::abs(dx);
    dy = std::abs(dy);

    int xx, xy, yx, yy;
    if (dx > dy)
    {
        xx = xsign;
        xy = 0;
        yx = 0;
        yy = ysign;
    }
    else
    {
        std::swap(dx, dy);
        xx = 0;
        xy = ysign;
        yx = xsign;
        yy = 0;
    }

    int D = 2 * dy - dx;
    int y = 0;

    std::vector<std::pair<int, int>> lineCells;
    for (int x = 0; x < dx + 1; x++)
    {
        lineCells.emplace_back(x0 + x * xx + y * yx, y0 + x * xy + y * yy);
        if (D >= 0)
        {
            y += 1;
            D -= 2 * dx;
        }
        D += 2 * dy;
    }
    return lineCells;
}

void OccupancyGrid::UpdateCellVis(const int &col, const int &row, const Eigen::Vector3d &color)
{
    // int startIdx = col + row * NUM_COLS_;
    // Set the color associated to 4 vertices associated to a given cell
    for (size_t k = 0; k < 4; k++)
    {
        // triangleMesh_->vertex_colors_.at(startIdx + k) = color;
        triangleMesh_->vertex_colors_.emplace_back(color);
    }
}

void OccupancyGrid::UpdateGridVis()
{
    static Eigen::Vector3d occupiedColor(1.0, 0.0, 0.0);
    static Eigen::Vector3d freeColor(1.0, 1.0, 1.0);
    static Eigen::Vector3d unknownColor(0.5, 0.5, 0.5);

    // Reset all grid color
    // NOTE: Reassigning colors to cells caused a weird behavior in Open3d. Hence we reset ...
    triangleMesh_->vertex_colors_.clear();

    // Update cell color based on grid occupancy
    for (int row = 0; row < NUM_ROWS_; row++)
    {
        for (int col = 0; col < NUM_COLS_; col++)
        {
            size_t cellIdx = row * NUM_COLS_ + col;
            if (grid_.probOcc.at(cellIdx) > 0.7)
            {
                // we use x-forward, y-left, o3d uses x-right, y-forward
                UpdateCellVis((NUM_COLS_ - 1) - col, row, occupiedColor);
            }
            else if (grid_.probOcc.at(cellIdx) < 0.3)
            {
                // we use x-forward, y-left, o3d uses x-right, y-forward
                UpdateCellVis((NUM_COLS_ - 1) - col, row, freeColor);
            }
            else
            {
                // we use x-forward, y-left, o3d uses x-right, y-forward
                UpdateCellVis((NUM_COLS_ - 1) - col, row, unknownColor);
            }
        }
    }
}

void OccupancyGrid::ShowGrid()
{
    // PrintGrid();
    UpdateGridVis();

    gridViz_.UpdateGeometry(triangleMesh_);
    gridViz_.PollEvents();
    gridViz_.UpdateRender();
}