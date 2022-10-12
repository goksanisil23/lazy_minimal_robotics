#include <iostream>
#include <thread>

#include <pcl/common/io.h> // for copyPointCloud
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h> // for PointCloud
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std::chrono_literals;

int main(int argc, char **argv)
{
    // initialize PointClouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);

    if (argc == 1)

    {
        // populate our PointCloud with points
        cloud->width    = 500;
        cloud->height   = 1;
        cloud->is_dense = false;
        cloud->points.resize(cloud->width * cloud->height);
        for (size_t i = 0; i < static_cast<size_t>(cloud->size()); ++i)
        {

            (*cloud)[i].x = 1024 * rand() / (RAND_MAX + 1.0);
            (*cloud)[i].y = 1024 * rand() / (RAND_MAX + 1.0);
            if (i % 2 == 0)
                (*cloud)[i].z = 1024 * rand() / (RAND_MAX + 1.0);
            else
                (*cloud)[i].z = -1 * ((*cloud)[i].x + (*cloud)[i].y);
        }
    }
    else if (argc == 2)
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) == -1)
        {
            std::cout << "Cloud reading failed." << std::endl;
            return (-1);
        }
    }
    else
    {
        std::cerr << "more than 2 args not supported" << std::endl;
    }
    std::cout << "read " << cloud->size() << " points" << std::endl;

    std::vector<int> inliers;

    // A) SampleConsensusModelPlane
    // {
    //     // created RandomSampleConsensus object and compute the appropriated model
    //     pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr plane_model(
    //         new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud));

    //     pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(plane_model);
    //     ransac.setDistanceThreshold(.01);
    //     ransac.computeModel();
    //     ransac.getInliers(inliers);

    //     // copies all inliers of the model computed to another PointCloud
    //     pcl::copyPointCloud(*cloud, inliers, *final);
    //     std::cout << "num plane inliers: " << inliers.size() << std::endl;
    // }

    // B) SACSegmentation
    {
        pcl::PointIndices::Ptr              inlierIndices(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr         coefficients(new pcl::ModelCoefficients);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true); // re-adjusts the coefficients after inlier is found
        seg.setModelType(pcl::SACMODEL_PLANE);
        // this is the distance of the point to the plane model (not the distance of the point to some neighbor)
        seg.setDistanceThreshold(0.1);
        seg.setInputCloud(cloud);
        seg.segment(*inlierIndices, *coefficients);
        if (inlierIndices->indices.size() == 0)
        {
            PCL_ERROR("Could not estimate a planar model for the given dataset.\n");
            return -1;
        }
        else
        {
            pcl::copyPointCloud(*cloud, inlierIndices->indices, *final);
        }
        std::cout << "num plane inliers: " << inlierIndices->indices.size() << std::endl;
    }

    // ----- Visualization ----- //
    pcl::visualization::PCLVisualizer viewer;
    viewer.setBackgroundColor(0, 0, 0);
    pcl::RGB rgb_plane    = pcl::GlasbeyLUT::at(1); // unique color
    pcl::RGB rgb_original = pcl::GlasbeyLUT::at(2); // unique color
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colour_handle_plane(
        final, rgb_plane.r, rgb_plane.g, rgb_plane.b); // Create colour handle
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> colour_handle_original(
        cloud, rgb_original.r, rgb_original.g, rgb_original.b); // Create colour handle
    viewer.addPointCloud<pcl::PointXYZ>(cloud, colour_handle_original, "cloud_original");
    viewer.addPointCloud<pcl::PointXYZ>(final, colour_handle_plane, "cloud_plane");
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        std::this_thread::sleep_for(10ms);
    }

    return 0;
}
