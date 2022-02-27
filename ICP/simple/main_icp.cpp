#include <iostream>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

Eigen::Matrix3d computeCrossCovar(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, const std::vector<int16_t>& correspondences)
{
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for(int16_t i = 0; i < input.cols(); i++)
    {
        auto input_pt = input.col(i);
        auto target_pt = target.col(correspondences.at(i));
        H += input_pt * target_pt.transpose();
    }
    return H;
}

void findCorrespondences(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, std::vector<int16_t>& correspondences)
{
    // For each point in the input set, find the closest one in target set
    for(auto input_itr = input.colwise().begin(); input_itr != input.colwise().end(); input_itr++) 
    {
        double min_dist = std::numeric_limits<double>::max();
        int16_t chosen_idx = -1;
        for(auto target_itr = target.colwise().begin(); target_itr != target.colwise().end(); target_itr++)
        {
            double dist = (*input_itr - *target_itr).norm();
            if(dist < min_dist){
                min_dist = dist;
                chosen_idx = target_itr - target.colwise().begin();
            }
        }
        correspondences.at(input_itr - input.colwise().begin()) = chosen_idx;
    } 
}

void drawCorrespondences(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, const std::vector<int16_t>& correspondences)
{
    // plt::clf();
    plt::subplot2grid(2, 1, 1, 0);
    plt::title("Correspondences");
    std::vector<double> input_x, input_y, target_x, target_y;
    // plot the input and target
    for(int i = 0; i < input.cols(); i++)
    {
        input_x.push_back(input(0,i));
        input_y.push_back(input(1,i));
        target_x.push_back(target(0,i));
        target_y.push_back(target(1,i));        
    }

    plt::named_plot("input", input_x, input_y, "--o");
    plt::named_plot("target", target_x, target_y, "--o");
    plt::grid(true);
    plt::legend();    

    // plot the correspondences
    for(int16_t i = 0; i < input.cols(); i++)
    {
        auto input_pt = input.col(i);
        auto target_pt = target.col(correspondences.at(i));
        plt::plot(std::vector<double>{input_pt(0), target_pt(0)}, 
                  std::vector<double>{input_pt(1), target_pt(1)}
                );
    }

    plt::pause(3.5);
}

void drawAction(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target)
{
    plt::clf();
    plt::subplot2grid(2, 1, 0, 0);
    plt::title("Global Registration");

    std::vector<double> input_x, input_y, target_x, target_y;
    // plot the input and target
    for(int i = 0; i < input.cols(); i++)
    {
        input_x.push_back(input(0,i));
        input_y.push_back(input(1,i));
        target_x.push_back(target(0,i));
        target_y.push_back(target(1,i));        
    }
    
    plt::named_plot("input", input_x, input_y, "--o");
    plt::named_plot("target", target_x, target_y, "--o");
    plt::grid(true);
    plt::legend();
    plt::xlim(-10, 30);
    plt::ylim(-5, 30);
    // plt::pause(2);
    
}

Eigen::Matrix4d rigidTransform3D(Eigen::MatrixXd input, Eigen::MatrixXd target)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d T;
    Eigen::Matrix4d Trans_total = Eigen::Matrix4d::Identity(); // accumulated homogenous transformation 
    Eigen::Vector3d centroid_target = target.rowwise().mean();
    Eigen::MatrixXd target_centered =  target.colwise() - centroid_target;
    const int16_t max_iterations = 30;
    std::vector<int16_t> correspondences(input.cols()); // same order as the input container : input -> target
    std::vector<int16_t> prev_correspond(correspondences); // same order as the input container : input -> target

    for(int i = 0; i < max_iterations; i++)
    {
        // Show the action
        drawAction(input, target);

        // 1) Center the point sets
        Eigen::Vector3d centroid_input = input.rowwise().mean();
        Eigen::MatrixXd input_centered =  input.colwise() - centroid_input;

        // 2) Find the closest correspondences between input and target
        findCorrespondences(input_centered, target_centered, correspondences);
        drawCorrespondences(input_centered, target_centered, correspondences);

        // 3) Compute the covariance matrix H
        Eigen::Matrix3d H = computeCrossCovar(input_centered, target_centered, correspondences);

        // 4) Find the rotation via SVD
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
        // Handle the special reflection case
        double d = (svd.matrixV() * svd.matrixU().transpose()).determinant();
        d = (d > 0) ? 1 : -1;
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity(3,3);
        I(2,2) = d;
        R = svd.matrixV() * I * svd.matrixU().transpose();
        T = -R * centroid_input + centroid_target;
        // Accumulate this iteration's transform into global transform
        Eigen::Matrix4d Trans = Eigen::Matrix4d::Identity(); 
        Trans.block<3,3>(0,0) = R;
        Trans.block<3,1>(0,3) = T;
        Trans_total =  Trans * Trans_total; // since R is left multiplier

        // 5) Apply the rotation to the last uncentered input
        input = (R * input).colwise() + T;
        
        // 6) Check if the correspondences have converged
        if( correspondences != prev_correspond) {
            prev_correspond = correspondences;
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

int main() 
{
    int8_t N_points = 30;
    Eigen::MatrixXd input = Eigen::MatrixXd(3, N_points);
    for(int i = 0; i < N_points; i++) {
        input(0, i) = i;
        input(1, i) = 0.2*i*(std::tan(0.5*i)*std::cos(0.5*i));
        input(2, i) = 0; // 2D for visualization
    }
    Eigen::MatrixXd target(3, N_points);
    Eigen::Quaternion<double> Q(3,0,0,1);
    Q.normalize();
    Eigen::Matrix3d R = Q.toRotationMatrix();
    Eigen::Vector3d T(-2,5,0);

    std::cout << "R_true:\n" << R << "\nT_true:\n" << T << std::endl;

    target = (R * input).colwise()  + T;

    plt::figure_size(800, 800);
    Eigen::Matrix4d trans = rigidTransform3D(input, target);
    std::cout << "R_ICP:\n" << trans.block<3,3>(0,0) << "\nT_ICP:\n" << trans.block<3,1>(0,3) << std::endl;

    return 0;
}
