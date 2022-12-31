#include "OccupancyGrid.h"

#include <algorithm>
#include <functional>
#include <math.h>
#include <random>

#include "TimeUtil.h"

OccupancyGrid::OccupancyGrid(const int   &numRows,
                             const int   &numCols,
                             const float &resolution,
                             const float &alpha,
                             const float &beta)
    : NUM_ROWS_{numRows}, NUM_COLS_{numCols}, GRID_RESOLUTION_{resolution}, ALPHA_{alpha}, BETA_{beta}, SENSOR_RANGE_{
                                                                                                            100.0f}
{
    assert(NUM_COLS_ % 2 == 0);

    InitGrid();
    InitVisualizer();
}

void OccupancyGrid::InitGrid()
{
    std::vector<Cell> colCells(NUM_ROWS_, Cell());
    grid_.resize(NUM_COLS_, colCells);

    // // X-forward, y-left. Grid(0,0) at bottom right
    static float GRID_ORIGO_Y = -(float)(NUM_COLS_ / 2) * GRID_RESOLUTION_;
    static float GRID_ORIGO_X = 0.0f;
    // sensor position in grid coordinate system.
    // Note that cell(0,0) does NOT correspond to x=0,y=0, since we center the grid at the sensor
    // But the cell indices still start from bottom right
    static float SENSOR_POS_X = 0.0f;
    static float SENSOR_POS_Y = 0.0f;

    for (int row = 0; row < NUM_ROWS_; row++)
    {
        for (int col = 0; col < NUM_COLS_; col++)
        {
            grid_.at(col).at(row).centerX = (float)row * GRID_RESOLUTION_ + GRID_ORIGO_X;
            grid_.at(col).at(row).centerY = (float)col * GRID_RESOLUTION_ + GRID_ORIGO_Y;
            grid_.at(col).at(row).range   = std::sqrt(std::pow(grid_.at(col).at(row).centerY - SENSOR_POS_Y, 2) +
                                                    std::pow(grid_.at(col).at(row).centerX - SENSOR_POS_X, 2));
            grid_.at(col).at(row).bearing = std::fmod((std::atan2(grid_.at(col).at(row).centerY - SENSOR_POS_Y,
                                                                  grid_.at(col).at(row).centerX - SENSOR_POS_X)) +
                                                          M_PI,
                                                      2.0 * M_PI) -
                                            M_PI;
        }
    }
}

void OccupancyGrid::InitVisualizer()
{
    gridViz_.CreateVisualizerWindow("Grid", 720, 1280, 1700, 270);
    gridViz_.GetRenderOption().background_color_      = {0.05, 0.05, 0.05};
    gridViz_.GetRenderOption().point_size_            = 1;
    gridViz_.GetRenderOption().show_coordinate_frame_ = true;

    // Eigen::Vector3i voxelIdx(col, row, 0);

    triangleMesh_ = std::make_shared<open3d::geometry::TriangleMesh>();
    for (int row = 0; row < NUM_ROWS_; row++)
    {
        for (int col = 0; col < NUM_COLS_; col++)
        {
            // Create a quad for each grid cell
            std::vector<Eigen::Vector3d> vertices;
            vertices.emplace_back((double)((NUM_COLS_ - 1) - col) * (double)GRID_RESOLUTION_,
                                  (double)row * (double)GRID_RESOLUTION_,
                                  0.0);
            vertices.emplace_back((double)((NUM_COLS_ - 1) - col + 1) * (double)GRID_RESOLUTION_,
                                  (double)row * (double)GRID_RESOLUTION_,
                                  0.0);
            vertices.emplace_back((double)((NUM_COLS_ - 1) - col + 1) * (double)GRID_RESOLUTION_,
                                  (double)(row + 1) * (double)GRID_RESOLUTION_,
                                  0.0);
            vertices.emplace_back((double)((NUM_COLS_ - 1) - col) * (double)GRID_RESOLUTION_,
                                  (double)(row + 1) * (double)GRID_RESOLUTION_,
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

void OccupancyGrid::UpdateGrid(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr)
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
            // Find the sensor measurement (ray) that is closest in angle to this cell
            size_t k{FindClosestBearingRayToCell(grid_.at(col).at(row).bearing, sensorRayBearings)};

            // Check:
            // If cell distance greater than maximum sensor range
            // If cell distance is behind the associated sensor ray + ALPHA_region
            // If cell is outside the field of view of the associated sensor ray + BETA_REGION
            if ((grid_.at(col).at(row).range > std::fmin(SENSOR_RANGE_, sensorRayRanges.at(k) + ALPHA_ / 2.0f)) ||
                ((std::fabs(grid_.at(col).at(row).bearing - sensorRayBearings.at(k)) > (BETA_ / 2.0f))))
            {
                grid_.at(col).at(row).probOcc = 0.5f;
            }
            // If sensor ray measurement lies within this cell (+ALPHA region) = OCCUPIED
            else if ((sensorRayRanges.at(k) < SENSOR_RANGE_) &&
                     (std::fabs(sensorRayRanges.at(k) - grid_.at(col).at(row).range) < (ALPHA_ / 2.0f)))
            {
                grid_.at(col).at(row).probOcc = 0.75f;
            }
            // If the sensor ray measurement is behind the cell = UNOCCUPIED
            else if (sensorRayRanges.at(k) > grid_.at(col).at(row).range)
            {
                grid_.at(col).at(row).probOcc = 0.25f;
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

    // auto lineCelss{BresenhamLineCells(0, 10, 0, 30)};
    // for (const auto &cell : lineCelss)
    // {
    //     grid_.at(cell.second).at(cell.first) = 1.0;
    // }
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
            if (grid_.at(col).at(row).probOcc > 0.7)
            {
                // we use x-forward, y-left, o3d uses x-right, y-forward
                UpdateCellVis((NUM_COLS_ - 1) - col, row, occupiedColor);
            }
            else if (grid_.at(col).at(row).probOcc < 0.3)
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