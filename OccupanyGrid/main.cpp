
#include "CarlaSim.h"

#include "OccupancyGrid.h"

constexpr u_int32_t NUM_SIM_STEPS = 500;

int main()
{
    CarlaSim carlaSim;
    try
    {
        // Setup the simulator
        carlaSim.Setup();
        boost::shared_ptr<cc::Vehicle> vehicle = carlaSim.SpawnVehicle();
        boost::shared_ptr<cc::Sensor>  lidar   = carlaSim.SpawnLidar();
        // Register a callback to listen to lidar data.
        lidar->Listen(std::bind(&CarlaSim::LidarCallback, &carlaSim, std::placeholders::_1));
        u_int32_t simCtr = 0;

        // Setup the Occupancy Grid
        OccupancyGrid oGrid(100.0, 100.0, 1.0);

        while (simCtr < NUM_SIM_STEPS)
        {
            // Iterate the sim
            carlaSim.Step();
            carlaSim.MoveSpectator();
            boost::shared_ptr<SemanticLidarData> lidarDataPtr;
            Eigen::Matrix4f                      lidarPose;
            carlaSim.GetLidarData(lidarDataPtr);
            carlaSim.GetLidarPose(lidarPose);
            carlaSim.UpdateCloudViz(lidarDataPtr);
            simCtr++;

            // Update the occupancy grid
            oGrid.UpdateGrid(lidarDataPtr);

            oGrid.ShowGrid();

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        carlaSim.Terminate();
        std::cout << "Finished successfully\n";
    }
    catch (const cc::TimeoutException &e)
    {
        carlaSim.Terminate();
        std::cout << '\n' << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        carlaSim.Terminate();
        std::cout << "\nException: " << e.what() << std::endl;
        return 2;
    }
}
