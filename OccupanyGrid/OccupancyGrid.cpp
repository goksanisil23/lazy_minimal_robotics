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
      SENSOR_RANGE_{100.0f}, SENSOR_POS_X_{0.0f}, SENSOR_POS_Y_{0.0f}, THRESH_P_OCCUPIED_{0.7}, THRESH_P_FREE_{0.3}
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
            grid_.centerX.at(cellIdx) = (float)row * GRID_RESOLUTION_ + GRID_ORIGO_X + GRID_RESOLUTION_ / 2.0;
            grid_.centerY.at(cellIdx) = (float)col * GRID_RESOLUTION_ + GRID_ORIGO_Y + GRID_RESOLUTION_ / 2.0;
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
    gridViz_.CreateVisualizerWindow("Grid", 720, 1440, 900, 270);
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

// Ignore returns from ground and higher than sensor, and calculate the range and bearing for those.
void OccupancyGrid::FilterAndCalculateSensorRayRangeAndBearings(
    const boost::shared_ptr<SemanticLidarData> &pointcloudPtr,
    std::vector<float>                         &sensorRayBearings,
    std::vector<float>                         &sensorRayRanges)
{
    for (const auto &pt : *pointcloudPtr)
    {
        if (!IsGroundHit(pt) && (IsAboveSensor(pt)))
        {
            sensorRayRanges.emplace_back(std::sqrt(pt.point.y * pt.point.y + pt.point.x * pt.point.x));
            sensorRayBearings.emplace_back(std::fmod((std::atan2(pt.point.y, pt.point.x)) + M_PI, 2.0 * M_PI) - M_PI);
        }
    }
}

void OccupancyGrid::UpdateGridNaive(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr)
{
    std::vector<float> sensorRayBearings;
    std::vector<float> sensorRayRanges;

    auto t0 = time_util::chronoNow();
    FilterAndCalculateSensorRayRangeAndBearings(pointcloudPtr, sensorRayBearings, sensorRayRanges);
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

inline bool OccupancyGrid::IsIndexWithinGrid(const int &rowIdx, const int &colIdx)
{
    return ((rowIdx >= 0) && (rowIdx < NUM_ROWS_) && (colIdx >= 0) && (colIdx < NUM_COLS_));
}

// Discretize the point coordinates in space to grid cell indices, check if point lies within the grid
// Returns (row,col,isWithinGrid)
std::tuple<int, int, bool> OccupancyGrid::DiscretizePointToCell(const float &x, const float &y)
{
    int gridColIdx = std::floor(y / GRID_RESOLUTION_ + NUM_COLS_ / 2.0);
    int gridRowIdx = std::floor(x / GRID_RESOLUTION_);

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
        (hit.object_tag == static_cast<uint32_t>(carla::rpc::CityObjectLabel::RoadLines)) ||
        (hit.object_tag == static_cast<uint32_t>(carla::rpc::CityObjectLabel::Sidewalks)) ||
        (hit.object_tag == static_cast<uint32_t>(carla::rpc::CityObjectLabel::Ground)))
    {
        return true;
    }
    return false;
}

inline bool OccupancyGrid::IsAboveSensor(const carla::sensor::data::SemanticLidarDetection &hit)
{
    return hit.point.z > 0.0;
}

void OccupancyGrid::SortAndDiscretizePointloud(
    const boost::shared_ptr<SemanticLidarData>        &pointcloudPtr,
    std::vector<std::tuple<int, int, bool>>           &hitCells,
    std::multimap<float, size_t, std::greater<float>> &rangeSortedCloudIndices)
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

// The grid is reset every measurement and occupancy is re-assigned only based on current measurement.
void OccupancyGrid::UpdateGridBresenhamOneShot(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr)
{
    // Reset the grid status
    auto t0 = time_util::chronoNow();
    std::fill(grid_.probOcc.begin(), grid_.probOcc.end(), 0.5);
    auto t1 = time_util::chronoNow();

    // Sort the pointcloud based on range
    std::vector<std::tuple<int, int, bool>>           hitCells(pointcloudPtr->size());
    std::multimap<float, size_t, std::greater<float>> rangeSortedCloudIndices;
    SortAndDiscretizePointloud(pointcloudPtr, hitCells, rangeSortedCloudIndices);
    auto t2 = time_util::chronoNow();

    // For each lidar hit, we trace the ray from the sensor to the hit.
    // Along the ray, we update cells as free until the hit cell.
    // If we see a cell with high occupancy along the ray already, we stop the traversal (leaving rest of the cells at 0.5)
    // NOTE: if cells behind 1st occupied cell needs to be captured, sort multimap from high to low range instead
    for (const auto &el : rangeSortedCloudIndices)
    {
        size_t rangeSortedCloudIdx{el.second};

        const auto                       &hit{pointcloudPtr->at(rangeSortedCloudIdx)};
        const std::tuple<int, int, bool> &hitCell{hitCells.at(rangeSortedCloudIdx)};

        // If the cell coordinate is within bounding box AND below the sensor AND not ground
        if (std::get<2>(hitCell) && !IsAboveSensor(hit) && !IsGroundHit(hit))
        {
            auto cellsAlongRay{
                BresenhamLineCells(SENSOR_POS_ROW_, SENSOR_POS_COL_, std::get<0>(hitCell), std::get<1>(hitCell))};

            for (int i = 0; i < cellsAlongRay.size() - 1; i++) // not including the hit point cell
            {
                int cellIdx = cellsAlongRay.at(i).first * NUM_COLS_ + cellsAlongRay.at(i).second;
                if ((grid_.probOcc.at(cellIdx) > THRESH_P_OCCUPIED_))
                {
                    break;
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

inline float OccupancyGrid::Logit(const float &probability)
{
    return std::log(probability / (1.0f - probability));
}

inline float OccupancyGrid::RetrieveProb(const float &logit)
{
    return 1.0f - (1.0f / (1.0f + std::exp(logit)));
}

// Grid state is propagated to the next step via the robot pose.
void OccupancyGrid::UpdateGridBresenhamCumulative(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr,
                                                  const Eigen::Matrix4f                      &lidarPose)
{
    // Here we use "probOcc" field of the grid as "logit" values actually

    // Transform the previous grid with robot pose, to form a prior
    auto t0 = time_util::chronoNow();
    PropagateGrid(lidarPose);
    // std::fill(grid_.probOcc.begin(), grid_.probOcc.end(), 0.0); //Logit(0.5)
    auto t1 = time_util::chronoNow();

    // Sort the pointcloud based on range
    std::vector<std::tuple<int, int, bool>>           hitCells(pointcloudPtr->size());
    std::multimap<float, size_t, std::greater<float>> rangeSortedCloudIndices;
    SortAndDiscretizePointloud(pointcloudPtr, hitCells, rangeSortedCloudIndices);
    auto t2 = time_util::chronoNow();

    for (const auto &el : rangeSortedCloudIndices)
    {
        size_t rangeSortedCloudIdx{el.second};

        const auto                       &hit{pointcloudPtr->at(rangeSortedCloudIdx)};
        const std::tuple<int, int, bool> &hitCell{hitCells.at(rangeSortedCloudIdx)};

        // If the cell coordinate is within bounding box AND below the sensor AND not ground
        if (std::get<2>(hitCell) && !IsAboveSensor(hit) && !IsGroundHit(hit))
        {
            auto cellsAlongRay{
                BresenhamLineCells(SENSOR_POS_ROW_, SENSOR_POS_COL_, std::get<0>(hitCell), std::get<1>(hitCell))};

            for (int i = 0; i < cellsAlongRay.size() - 1; i++) // not including the hit point cell
            {
                int cellIdx = cellsAlongRay.at(i).first * NUM_COLS_ + cellsAlongRay.at(i).second;

                {
                    grid_.probOcc.at(cellIdx) += Logit(0.3);
                }
            }
            // For the last cell on the ray line, assign occupied since thats the hit cell
            int hitPtCellIdx = cellsAlongRay.back().first * NUM_COLS_ + cellsAlongRay.back().second;
            {
                grid_.probOcc.at(hitPtCellIdx) += Logit(0.9);
            }
        }
    }
    auto t3 = time_util::chronoNow();

    time_util::showTimeDuration(t1, t0, "reset grid: ");
    time_util::showTimeDuration(t2, t1, "sort/disc :  ");
    time_util::showTimeDuration(t3, t2, "occloop   : ");
}

// Transform the grid cells with the provided pose.
// Eliminates the cells that falls outside the grid boundaries.
// Cells that are not refilled, are initialized to 0.5 probability.
void OccupancyGrid::PropagateGrid(const Eigen::Matrix4f &lidarPose)
{
    std::vector<float> newProbOcc(grid_.probOcc.size(), 0.0); //Logit(0.5)
    std::vector<float> transformedCenterXs;
    std::vector<float> transformedCenterYs;

    static Eigen::Matrix4f prevLidarPose{lidarPose};
    // how much lidar has moved w.r.t previous pose
    Eigen::Matrix4f T_lidar_k_to_kplus{prevLidarPose.inverse() * lidarPose};
    Eigen::Matrix4f T_grid{T_lidar_k_to_kplus.inverse()};
    std::cout << "T_grid\n" << T_grid << std::endl;
    // Transform the grid with the inverse relative lidar movement
    // ------------------- A ------------------
    // 1) Transform the cell centers with T_grid
    // 2) Discretize the transformed grid centers, to find new coordinates
    // 3) If within grid, Carry the old probability to the new coordinates
    // ------------------- B ------------------
    // 1) Transform the cell centers with T_grid
    // 2) Find how much center has moved in 2D. (delta)
    // 3) Discretize the (delta) and add it to previous cell coordinate
    // 4) If within grid, Carry the old probability to the new coordinates
    for (int row = 0; row < NUM_ROWS_; row++)
    {
        for (int col = 0; col < NUM_COLS_; col++)
        {
            const size_t oldCellIdx = row * NUM_COLS_ + col;

            Eigen::Vector4f transformedCellCenter{
                T_grid * Eigen::Vector4f{grid_.centerX.at(oldCellIdx), grid_.centerY.at(oldCellIdx), 0.0f, 1.0f}};

            // A)
            // const float deltaX{transformedCellCenter.x() - grid_.centerX.at(oldCellIdx)};
            // const float deltaY{transformedCellCenter.y() - grid_.centerY.at(oldCellIdx)};
            // const int   deltaRow{static_cast<int>(deltaX / GRID_RESOLUTION_)};
            // const int   deltaCol{static_cast<int>(deltaY / GRID_RESOLUTION_)};
            // // std::cout << "(" << deltaRow << "," << deltaCol << ")";
            // if (IsIndexWithinGrid(row + deltaRow, col + deltaCol))
            // {
            //     const size_t newCellIdx = (row + deltaRow) * NUM_COLS_ + (col + deltaCol);
            //     // Carry the probability
            //     newProbOcc.at(newCellIdx) = grid_.probOcc.at(oldCellIdx);
            // }

            // B)
            std::tuple<int, int, bool> transformedCellCoords{
                DiscretizePointToCell(transformedCellCenter.x(), transformedCellCenter.y())};
            // If transformed cell center lies within the grid
            if (std::get<2>(transformedCellCoords))
            {
                const size_t newCellIdx =
                    std::get<0>(transformedCellCoords) * NUM_COLS_ + std::get<1>(transformedCellCoords);
                // Carry the probability
                newProbOcc.at(newCellIdx) = grid_.probOcc.at(oldCellIdx);
            }
        }
    }
    // std::cout << "\n";

    grid_.probOcc = newProbOcc;
    prevLidarPose = lidarPose;
}

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
            if (grid_.probOcc.at(cellIdx) > THRESH_P_OCCUPIED_)
            {
                // we use x-forward, y-left, o3d uses x-right, y-forward
                UpdateCellVis((NUM_COLS_ - 1) - col, row, occupiedColor);
            }
            else if (grid_.probOcc.at(cellIdx) < THRESH_P_FREE_)
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

void OccupancyGrid::UpdateCellVisLogit(const int &col, const int &row, const float &prob)
{
    // Set the color associated to 4 vertices associated to a given cell
    for (size_t k = 0; k < 4; k++)
    {
        // triangleMesh_->vertex_colors_.at(startIdx + k) = color;
        triangleMesh_->vertex_colors_.emplace_back(Eigen::Vector3d(1.0 - prob, 1.0 - prob, 1.0 - prob));
    }
}

void OccupancyGrid::UpdateGridVisLogit()
{

    // Reset all grid color
    // NOTE: Reassigning colors to cells caused a weird behavior in Open3d. Hence we reset ...
    triangleMesh_->vertex_colors_.clear();

    // Update cell color based on grid occupancy
    for (int row = 0; row < NUM_ROWS_; row++)
    {
        for (int col = 0; col < NUM_COLS_; col++)
        {
            size_t cellIdx = row * NUM_COLS_ + col;
            float  prob{RetrieveProb(grid_.probOcc.at(cellIdx))};

            // we use x-forward, y-left, o3d uses x-right, y-forward
            UpdateCellVisLogit((NUM_COLS_ - 1) - col, row, prob);
        }
    }
}

void OccupancyGrid::ShowGrid()
{
    // PrintGrid();
    // UpdateGridVis();
    UpdateGridVisLogit();

    gridViz_.UpdateGeometry(triangleMesh_);
    gridViz_.GetViewControl().SetZoom(0.52);
    gridViz_.PollEvents();
    gridViz_.UpdateRender();
}