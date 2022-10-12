#include <iostream>
#include <vector>
#include <chrono>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter_indices.h> // for pcl::removeNaNFromPointCloud
#include <pcl/segmentation/region_growing.h>

namespace my_util
{
    constexpr auto &chronoNow = std::chrono::high_resolution_clock::now;
    using time_point = decltype(std::chrono::high_resolution_clock::now());
}

void showTimeDuration(const my_util::time_point &t2, const my_util::time_point &t1, const std::string &message)
{
    std::cout << message << std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count() << std::endl;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "provide pcd file" << std::endl;
        return -1;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) == -1)
    {
        std::cout << "Cloud reading failed." << std::endl;
        return (-1);
    }

    std::cout << "number of points in cloud: " << cloud->size() << std::endl;

    auto t1 = my_util::chronoNow();

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;

    normal_estimator.setInputCloud(cloud);
    normal_estimator.setSearchMethod(tree);
    // normal_estimator.setKSearch(50);
    normal_estimator.setRadiusSearch(0.03);
    normal_estimator.compute(*normals);

    auto t2 = my_util::chronoNow();

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::removeNaNFromPointCloud(*cloud, *indices);

    auto t3 = my_util::chronoNow();

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(50);
    reg.setMaxClusterSize(1000000);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(30);
    reg.setInputCloud(cloud);
    reg.setIndices(indices);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(1.0);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    auto t4 = my_util::chronoNow();

    showTimeDuration(t2, t1, "normal est: ");
    showTimeDuration(t3, t2, "nan remove: ");
    showTimeDuration(t4, t3, "region grw: ");

    std::cout << "Number of clusters: " << clusters.size() << std::endl;
    std::cout << "First cluster has " << clusters[0].indices.size() << " points." << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    pcl::visualization::CloudViewer viewer("Cluster viewer");
    viewer.showCloud(colored_cloud);
    while (!viewer.wasStopped())
    {
    }

    return (0);
}
