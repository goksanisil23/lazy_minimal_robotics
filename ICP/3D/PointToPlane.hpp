#pragma once

#include "ICPBase.hpp"

namespace ICP {

class PointToPlane : public ICPBase 
{
public:
    PointToPlane(int16_t max_iterations_in, bool visualize_in) 
        : ICPBase {max_iterations_in, visualize_in}
        {}


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

};
} // namespace ICP