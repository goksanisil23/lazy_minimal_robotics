
#include "CarlaSim.h"
#include "OccupancyGrid.h"
#include "TimeUtil.h"

constexpr u_int32_t NUM_SIM_STEPS = 5000;

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
        OccupancyGrid oGrid(400, 200, 0.25, 0.1, 0.05); // long,lat,res

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
            auto t0 = time_util::chronoNow();
            // oGrid.UpdateGridNaive(lidarDataPtr);
            oGrid.UpdateGridBresenham(lidarDataPtr);
            auto t1 = time_util::chronoNow();
            oGrid.ShowGrid();
            auto t2 = time_util::chronoNow();

            time_util::showTimeDuration(t1, t0, "update: ");
            time_util::showTimeDuration(t2, t1, "show  : ");

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            // getchar();
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
