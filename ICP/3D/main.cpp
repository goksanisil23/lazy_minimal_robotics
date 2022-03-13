#include <chrono>
#include "argparse.hpp"
#include "PointToPoint.hpp"
#include "PointToPlane.hpp"

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
    program.add_argument("-plane")
        .help("use Point-to-Plane algorithm")
        .default_value(false).implicit_value(true);                  
    program.parse_args(argc, argv);

    std::vector<Eigen::Vector3f> input;
    std::vector<Eigen::Vector3f> target;
    uint32_t N_points;
    // SAMPLE 1: 3D, more points
    if(!program.get<bool>("-2d")) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile<pcl::PointXYZ> (program.get<std::string>("--input_pcd"), *input_cloud);
        N_points = input_cloud->size();
        input.resize(N_points);
        target.resize(N_points);
        for(int i = 0; i < N_points; i++ ) {
            input.at(i) = {input_cloud->points.at(i).x,input_cloud->points.at(i).y,input_cloud->points.at(i).z};
        }

    }
    // SAMPLE 2: 2D, fewer points
    else{
        N_points = 30;
        // Create dummy input set
        input.resize(N_points);
        target.resize(N_points);
        for(int i = 0; i < N_points; i++) {
            float i_f = static_cast<float>(i);
            input.at(i) = {i_f, 0.2f*i_f*(std::tan(0.5f*i_f)*std::cos(0.5f*i_f)), 0}; // 2D for visualization
        }
    }

    // Define a rigid body translation + rotation
    // Eigen::Quaternion<float> Q(0,0,0,1);
    Eigen::Quaternion<float> Q(3,0,0,1); // w,x,y,z
    Q.normalize();
    Eigen::Matrix3f R = Q.toRotationMatrix();
    Eigen::Vector3f T(-2,5,0);

    // Map to matrix form, for easy transformation
    auto input_matrix = Eigen::Map<Eigen::MatrixXf>(input.at(0).data(), 3, input.size());
    
    // Target point set that we want to find rigid body alignment to.
    auto target_matrix = (R * input_matrix).colwise()  + T;

    // Map back to vector form
    Eigen::Matrix<float, 3, Eigen::Dynamic>::Map(target.at(0).data(), target_matrix.rows(), target_matrix.cols()) = target_matrix;
    
    std::cout << "R_true:\n" << R << "\nT_true:\n" << T << std::endl;

    // Create an ICP object and execute the alignment
    std::shared_ptr<ICP::ICPBase> lazyICP;
    if(!program.get<bool>("-plane"))
        lazyICP = std::make_shared<ICP::PointToPoint>(30, program.get<bool>("-visualize") ); // max_iterations, visualize
    else
        lazyICP = std::make_shared<ICP::PointToPlane>(30, program.get<bool>("-visualize") ); // max_iterations, visualize

    auto start_time = std::chrono::high_resolution_clock::now();
    Eigen::Matrix4f trans(lazyICP->rigidTransform3D(input_matrix, target_matrix));
    auto end_time = std::chrono::high_resolution_clock::now();
    auto delta_time = std::chrono::duration<float, std::chrono::seconds::period>(end_time-start_time).count();
    
    std::cout << "R_ICP:\n" << trans.block<3,3>(0,0) << "\nT_ICP:\n" << trans.block<3,1>(0,3) << std::endl;
    std::cout << "Took " << delta_time << " seconds" << std::endl;

    return 0;
}