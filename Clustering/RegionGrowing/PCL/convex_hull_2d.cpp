#include <thread>

#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h> // for copyPointCloud

#include <pcl/filters/project_inliers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std::chrono_literals;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "provide pcd file" << std::endl;
        return -1;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) == -1)
    {
        std::cout << "Cloud reading failed." << std::endl;
        return (-1);
    }

    // Plane segmentation
    pcl::ModelCoefficients::Ptr         coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr              inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    pcl::copyPointCloud(*cloud, inliers->indices, *cloud_filtered);

    // Project the model inliers (since actually the inliers do not perfectly lie on the plane equation)
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(cloud_filtered);
    proj.setIndices(inliers);
    proj.setModelCoefficients(coefficients);
    proj.filter(*cloud_projected);

    // Create a Convex Hull representation of the projected inliers
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ConvexHull<pcl::PointXYZ>      chull;
    chull.setInputCloud(cloud_projected);
    chull.reconstruct(*cloud_hull);

    // ----- Visualization ----- //
    pcl::visualization::PCLVisualizer viewer;
    viewer.setBackgroundColor(0, 0, 0);
    pcl::RGB rgb_plane    = pcl::GlasbeyLUT::at(1); // unique color
    pcl::RGB rgb_original = pcl::GlasbeyLUT::at(2); // unique color
    pcl::RGB rgb_hull     = pcl::GlasbeyLUT::at(4); // unique color
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colour_handle_plane(
        cloud_filtered, rgb_plane.r, rgb_plane.g, rgb_plane.b); // Create colour handle
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colour_handle_original(
        cloud, rgb_original.r, rgb_original.g, rgb_original.b); // Create colour handle
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colour_handle_hull(
        cloud_hull, rgb_hull.r, rgb_hull.g, rgb_hull.b); // Create colour handle
    viewer.addPointCloud<pcl::PointXYZ>(cloud, colour_handle_original, "cloud_original");
    viewer.addPointCloud<pcl::PointXYZ>(cloud_filtered, colour_handle_plane, "cloud_plane");
    viewer.addPointCloud<pcl::PointXYZ>(cloud_hull, colour_handle_hull, "chull");
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        std::this_thread::sleep_for(10ms);
    }

    return (0);
}
