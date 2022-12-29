#include "OccupancyGrid.h"

#include <functional>

OccupancyGrid::OccupancyGrid(const int &numCellsX, const int &numCellsY, const float &resolution)
    : NUM_CELLS_X_{numCellsX}, NUM_CELLS_Y_{numCellsY}, GRID_RESOLUTION_{resolution}
{
    assert(NUM_CELLS_X_ % 2 == 0);
    std::vector<float> initCellProbsRows(NUM_CELLS_Y_, 0.0);
    grid_.resize(NUM_CELLS_X_, initCellProbsRows);

    InitVisualizer();
}

void OccupancyGrid::InitVisualizer()
{
    gridViz_.CreateVisualizerWindow("Grid");

    // Create a 2D grid with 10 rows and 20 columns
    int                            rows = 10;
    int                            cols = 20;
    open3d::geometry::TriangleMesh mesh;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Create a triangle for each grid cell
            std::vector<Eigen::Vector3d> vertices;
            vertices.emplace_back((double)i, (double)j, 0.0);
            vertices.emplace_back((double)i + 1.0, (double)j, 0.0);
            vertices.emplace_back((double)i, (double)j + 1.0, 0.0);
            std::vector<Eigen::Vector3i> triangles;
            triangles.emplace_back(0, 1, 2);
            mesh.vertices_  = vertices;
            mesh.triangles_ = triangles;
            // Add the triangle to the mesh
            mesh.vertices_.insert(mesh.vertices_.end(), vertices.begin(), vertices.end());
            mesh.triangles_.insert(mesh.triangles_.end(), triangles.begin(), triangles.end());
        }
    }
    auto meshPtr = std::make_shared<open3d::geometry::TriangleMesh>(mesh);
    gridViz_.AddGeometry(meshPtr);
}

void OccupancyGrid::UpdateGrid(const boost::shared_ptr<SemanticLidarData> &pointcloudPtr)
{
    // Reset the grid
    std::vector<float> initCellProbsRows(NUM_CELLS_Y_, 0.0);
    grid_.resize(NUM_CELLS_X_, initCellProbsRows);

    for (const auto &pt : *pointcloudPtr)
    {
        int gridXidx = std::floor(pt.point.x / GRID_RESOLUTION_ + NUM_CELLS_X_ / 2);
        int gridYidx = std::floor(pt.point.y / GRID_RESOLUTION_);

        // Check if the point is within the grid
        if ((gridYidx >= 0) && (std::abs(gridYidx) < NUM_CELLS_Y_) && (gridXidx >= 0) && (gridXidx < NUM_CELLS_X_))
        {
            grid_.at(gridXidx).at(gridYidx) = 1.0;
        }
    }
}

void OccupancyGrid::ShowGrid()
{
    // gridViz_.UpdateGeometry(o3dCloud_);
    gridViz_.PollEvents();
    gridViz_.UpdateRender();
}