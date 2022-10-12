#include <iostream>
#include <thread>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h> // for PointCloud
#include <pcl/point_types.h>
#include <pcl/search/search.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/filters/radius_outlier_removal.h>

using namespace std::chrono_literals;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "provide pcd file" << std::endl;
        return -1;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) == -1)
    {
        std::cout << "Cloud reading failed." << std::endl;
        return (-1);
    }

    // Create a set of planar coefficients with X=Y=0,Z=1
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    coefficients->values.resize(4);
    coefficients->values[0] = 0;
    coefficients->values[1] = 1;
    coefficients->values[2] = 0;
    coefficients->values[3] = 0;

    // Create the filtering object
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud);
    proj.setModelCoefficients(coefficients);
    proj.filter(*cloud_projected);

    // ----- Visualization ----- //
    pcl::visualization::PCLVisualizer viewer;
    viewer.setBackgroundColor(0, 0, 0);
    pcl::RGB rgb_projected = pcl::GlasbeyLUT::at(1); // unique color
    pcl::RGB rgb_original  = pcl::GlasbeyLUT::at(2); // unique color
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colour_handle_projected(
        cloud_projected, rgb_projected.r, rgb_projected.g, rgb_projected.b); // Create colour handle
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colour_handle_original(
        cloud, rgb_original.r, rgb_original.g, rgb_original.b); // Create colour handle
    viewer.addPointCloud<pcl::PointXYZ>(cloud, colour_handle_original, "cloud_original");
    viewer.addPointCloud<pcl::PointXYZ>(cloud_projected, colour_handle_projected, "cloud_proj");
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        std::this_thread::sleep_for(10ms);
    }

    return (0);
}
