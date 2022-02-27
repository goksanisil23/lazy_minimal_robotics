#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>


std::pair<Eigen::Matrix3d, Eigen::Vector3d> rigidTransform3D(Eigen::MatrixXd input, Eigen::MatrixXd target)
{
    // Find the centroid of both set of points
    Eigen::Vector3d centroid_input = input.rowwise().mean();
    Eigen::Vector3d centroid_target = target.rowwise().mean();

    // Subtract the centroids from the point sets to center them at their origin
    Eigen::MatrixXd input_orig =  input.colwise() - centroid_input;
    Eigen::MatrixXd target_orig =  target.colwise() - centroid_target;

    // Compute the covariance matrix H
    Eigen::Matrix3d H = input_orig * target_orig.transpose(); // H = 3x3
    // Find the rotation via SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Handle the special reflection case
    double d = (svd.matrixV() * svd.matrixU().transpose()).determinant();
    d = (d > 0) ? 1 : -1;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity(3,3);
    I(2,2) = d;
    Eigen::Matrix3d R = svd.matrixV() * I * svd.matrixU().transpose();

    Eigen::Vector3d T = -R * centroid_input + centroid_target;

    return std::make_pair(R, T);
}


int main() 
{
    int8_t N_points = 100;
    Eigen::MatrixXd input = Eigen::MatrixXd::Random(3, N_points) * 10;
    Eigen::MatrixXd target(3, N_points);
    Eigen::Quaternion<double> Q(1,3,5,2);
    Q.normalize();
    Eigen::Matrix3d R = Q.toRotationMatrix();
    Eigen::Vector3d T = Eigen::MatrixXd::Random(3,1);

    target = (R * input).colwise()  + T;

    std::pair<Eigen::Matrix3d,Eigen::Vector3d> trans = rigidTransform3D(input, target);

    std::cout << R << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << trans.first << std::endl;

    std::cout << T << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << trans.second << std::endl;    


    // std::cout << target << std::endl;

    return 0;
}
