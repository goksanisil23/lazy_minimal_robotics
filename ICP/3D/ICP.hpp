#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "Mrpt.h"
#include <omp.h>

using namespace std::chrono_literals;

class ICP
{
public:
    ICP(int16_t max_iterations_in, bool visualize_in) : max_iterations(max_iterations_in), visualize(visualize_in),
                                                        input_cloud(new pcl::PointCloud<pcl::PointXYZ>),
                                                        target_cloud(new pcl::PointCloud<pcl::PointXYZ>)
    {
        if(visualize) {
            vis = std::make_shared<pcl::visualization::PCLVisualizer>("Vista 3f");
            vis->setBackgroundColor(0,0,0);

            vis->addPointCloud<pcl::PointXYZ>(input_cloud, std::string("input"));
            vis->addPointCloud<pcl::PointXYZ>(target_cloud, std::string("target"));
            vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, std::string("input"));
            vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, std::string("target"));
            vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, std::string("input"));
            vis->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, std::string("target"));  
        }
    }

    Eigen::Matrix3f computeCrossCovar(const Eigen::MatrixXf& input, const Eigen::MatrixXf& target, const std::vector<int16_t>& correspondences)
    {
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
        for(int16_t i = 0; i < input.cols(); i++)
        {
            auto input_pt = input.col(i);
            auto target_pt = target.col(correspondences.at(i));
            H += input_pt * target_pt.transpose();
        }
        return H;
    }

    void findCorrespondencesBruteForce(const Eigen::MatrixXf& input, const Eigen::MatrixXf& target, std::vector<int16_t>& correspondences)
    {
        // For each point in the input set, find the closest one in target set
        for(auto input_itr = input.colwise().begin(); input_itr != input.colwise().end(); input_itr++) 
        {
            float min_dist = std::numeric_limits<float>::max();
            int16_t chosen_idx = -1;
            for(auto target_itr = target.colwise().begin(); target_itr != target.colwise().end(); target_itr++)
            {
                float dist = (*input_itr - *target_itr).norm();
                if(dist < min_dist){
                    min_dist = dist;
                    chosen_idx = target_itr - target.colwise().begin();
                }
            }
            correspondences.at(input_itr - input.colwise().begin()) = chosen_idx;
        } 
    }

    void findCorrespondencesKnn(const Eigen::MatrixXf& input, const Eigen::MatrixXf& target, std::vector<int16_t>& correspondences)
    {   
        // For each point in the input set, find the closest one in target set
        for(auto input_itr = input.colwise().begin(); input_itr != input.colwise().end(); input_itr++) 
        {
            // we need the nearest neighbor, hence k = 1
            Eigen::VectorXi nn_indices(1);
            Mrpt::exact_knn(*input_itr, target, 1, nn_indices.data()); // query_point, target_dataset, num_nn_neighbors, indices_of_found_neighbors        
            correspondences.at(input_itr - input.colwise().begin()) = nn_indices(0);
        }
    }

    Eigen::Matrix4f rigidTransform3D(Eigen::MatrixXf input, Eigen::MatrixXf target)
    {
        Eigen::Matrix3f R;
        Eigen::Vector3f T;
        Eigen::Matrix4f Trans_total = Eigen::Matrix4f::Identity(); // accumulated homogenous transformation 
        Eigen::Vector3f centroid_target = target.rowwise().mean();
        Eigen::MatrixXf target_centered =  target.colwise() - centroid_target;
        std::vector<int16_t> correspondences(input.cols()); // same order as the input container : input -> target
        std::vector<int16_t> prev_correspond(correspondences); // same order as the input container : input -> target

        for(int i = 0; i < max_iterations; i++)
        {
            // 1) Center the point sets
            Eigen::Vector3f centroid_input = input.rowwise().mean();
            Eigen::MatrixXf input_centered =  input.colwise() - centroid_input;

            // 2) Find the closest correspondences between input and target
            // findCorrespondencesBruteForce(input_centered, target_centered, correspondences);
            findCorrespondencesKnn(input_centered, target_centered, correspondences);
            if(visualize) drawCorrespondences(input, target, correspondences);
            // if(visualize) drawCorrespondences(input_centered, target_centered, correspondences);

            // 3) Compute the covariance matrix H
            Eigen::Matrix3f H = computeCrossCovar(input_centered, target_centered, correspondences);

            // 4) Find the rotation via SVD
            Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
            // Handle the special reflection case
            float d = (svd.matrixV() * svd.matrixU().transpose()).determinant();
            d = (d > 0) ? 1 : -1;
            Eigen::Matrix3f I = Eigen::Matrix3f::Identity(3,3);
            I(2,2) = d;
            R = svd.matrixV() * I * svd.matrixU().transpose();
            T = -R * centroid_input + centroid_target;
            // Accumulate this iteration's transform into global transform
            Eigen::Matrix4f Trans = Eigen::Matrix4f::Identity(); 
            Trans.block<3,3>(0,0) = R;
            Trans.block<3,1>(0,3) = T;
            Trans_total =  Trans * Trans_total; // since R is left multiplier

            // 5) Apply the rotation to the last uncentered input
            input = (R * input).colwise() + T;
            
            // 6) Check if the correspondences have converged
            if( correspondences != prev_correspond) {
                prev_correspond = correspondences;
                std::cout << "iter: " << i << std::endl;
            }
            else {
                std::cout << "converged in " << i << " steps" << std::endl;
                return Trans_total;
                break;
            }

        }
        std::cout << "maximum iterations of " << max_iterations << " reached" << std::endl;
        return Trans_total;
    }

    // TODO: This is super slow now.
    void drawCorrespondences(const Eigen::MatrixXf& input, const Eigen::MatrixXf& target, const std::vector<int16_t>& correspondences)
    {

        vis->removeAllShapes();
        // vis->removeAllPointClouds();

        input_cloud->width = input.cols();
        target_cloud->width = input.cols();
        input_cloud->height = 1;
        target_cloud->height = 1;
        target_cloud->resize (target_cloud->width * target_cloud->height);
        input_cloud->resize (input_cloud->width * input_cloud->height);

        // plot the input and target
        for(int i = 0; i < input.cols(); i++)
        {
            // Draw target/source pointclouds
            input_cloud->points.at(i).x = input(0,i);
            input_cloud->points.at(i).y = input(1,i); 
            input_cloud->points.at(i).z = input(2,i);
            target_cloud->points.at(i).x = target(0,i);
            target_cloud->points.at(i).y = target(1,i);
            target_cloud->points.at(i).z = target(2,i);

            // Draw correspondence
            auto input_pt = input.col(i);
            auto target_pt = target.col(correspondences.at(i));
            vis->addLine<pcl::PointXYZ> (pcl::PointXYZ(input_pt(0),input_pt(1),input_pt(2)),
                                            pcl::PointXYZ(target_pt(0),target_pt(1),target_pt(2)), std::string("correspondence")+std::to_string(i));            
        }

        vis->updatePointCloud(input_cloud, std::string("input"));
        vis->updatePointCloud(target_cloud, std::string("target"));

        vis->spinOnce(100);
        // std::this_thread::sleep_for(2000ms);
    }

private:
    const int16_t max_iterations = 30;
    // Visualization related member vars
    bool visualize = true;
    std::shared_ptr<pcl::visualization::PCLVisualizer> vis;
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud;

};