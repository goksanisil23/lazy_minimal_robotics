#include <chrono>
#include "argparse.hpp"
#include "ICP.hpp"

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("ICP");
    program.add_argument("--input_pcd")
        .help("input pcd file");  
    program.add_argument("-visualize")
        .help("visualize ICP registration")
        .default_value(false).implicit_value(true);  
    program.add_argument("-2d")
        .help("use a simpler 2d curve")
        .default_value(false).implicit_value(true);          
    program.parse_args(argc, argv);

    Eigen::MatrixXf input;
    Eigen::MatrixXf target;
    // SAMPLE 1: 3D, more points
    if(!program.get<bool>("-2d")) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile<pcl::PointXYZ> (program.get<std::string>("--input_pcd"), *input_cloud);
        uint32_t N_points = input_cloud->size();
        input = Eigen::MatrixXf(3, N_points);
        target = Eigen::MatrixXf(3, N_points);
        for(int i = 0; i < N_points; i++ ) {
            input(0, i) = input_cloud->points.at(i).x;
            input(1, i) = input_cloud->points.at(i).y;
            input(2, i) = input_cloud->points.at(i).z;
        }

    }
    // SAMPLE 2: 2D, fewer points
    else{
        uint32_t N_points = 30;
        // Create dummy input set
        input = Eigen::MatrixXf(3, N_points);
        target = Eigen::MatrixXf(3, N_points);
        for(int i = 0; i < N_points; i++) {
            input(0, i) = i;
            input(1, i) = 0.2*i*(std::tan(0.5*i)*std::cos(0.5*i));
            input(2, i) = 0; // 2D for visualization
        }
    }

    // Define a rigid body translation + rotation
    // Eigen::Quaternion<float> Q(0,0,0,1);
    Eigen::Quaternion<float> Q(3,0,0,1);
    Q.normalize();
    Eigen::Matrix3f R = Q.toRotationMatrix();
    Eigen::Vector3f T(-2,5,0);

    std::cout << "R_true:\n" << R << "\nT_true:\n" << T << std::endl;

    // Target point set that we want to find rigid body alignment to.
    target = (R * input).colwise()  + T;

    // Create an ICP object and execute the alignment
    ICP icpNaive(30, program.get<bool>("-visualize") ); // max_iterations, visualize
    auto start_time = std::chrono::high_resolution_clock::now();
    Eigen::Matrix4f trans(icpNaive.rigidTransform3D(input, target));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto delta_time = std::chrono::duration<float, std::chrono::seconds::period>(end_time-start_time).count();
    std::cout << "R_ICP:\n" << trans.block<3,3>(0,0) << "\nT_ICP:\n" << trans.block<3,1>(0,3) << std::endl;
    std::cout << "Took " << delta_time << " seconds" << std::endl;


    return 0;
}