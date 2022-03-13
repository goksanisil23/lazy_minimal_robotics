#pragma once

#include "ICPBase.hpp"

namespace ICP {

class PointToPlane : public ICPBase 
{
public:
    PointToPlane(int16_t max_iterations_in, bool visualize_in) 
        : ICPBase {max_iterations_in, visualize_in}, N_NEIGHBORS_FOR_NORMALS{6}
        {}


    void surfaceNormalViaPCA(const Eigen::MatrixXf& surface_pts, Eigen::Vector3f& normal)
    {
        assert(surface_pts.cols() >= 3); // need to have at least 3 points for a surface
        Eigen::Vector3f surface_centroid = surface_pts.rowwise().mean();
        Eigen::MatrixXf surface_pts_centered = surface_pts.colwise() - surface_centroid;
        Eigen::Matrix3f surf_cov = surface_pts_centered * surface_pts_centered.transpose();

        // Get the eigen values, the smallest eigenvalue should be the normal
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig_solver; // since surf_cov is symmetric, we can use adjoint solver
        eig_solver.computeDirect(surf_cov); // faster than the compute() alternative
        normal = eig_solver.eigenvectors().col(0);
        // if(normal(2) > 0)
        //     normal = -normal;
    }

    void findNormals(const Eigen::MatrixXf& target, std::vector<Eigen::Vector3f>& target_normals)
    {
        // Find normals by considering the surface composed of N-nearest neighbors of a point
        for(auto target_pt_itr =  target.colwise().begin(); target_pt_itr != target.colwise().end(); target_pt_itr++)
        {
            // get N nearest neighbors
            std::vector<Eigen::Vector3f> surface_pts(N_NEIGHBORS_FOR_NORMALS);
            Eigen::VectorXi nn_indices(N_NEIGHBORS_FOR_NORMALS);
            Mrpt::exact_knn(*target_pt_itr, target, N_NEIGHBORS_FOR_NORMALS, nn_indices.data()) ; // query_point, target_dataset, num_nn_neighbors, indices_of_found_neighbors
            // Compute the normal from the surface these points represent
            surfaceNormalViaPCA(target(Eigen::all, nn_indices), target_normals.at(target_pt_itr - target.colwise().begin()));
        }

        // if(visualize) visualizeNormals(target, target_normals);
    }

    void visualizeNormals(const Eigen::MatrixXf& target, const std::vector<Eigen::Vector3f>& target_normals)
    {
        vis->removeAllShapes();
        // vis->removeAllPointClouds();

        target_cloud->width = target.cols();
        target_cloud->height = 1;
        target_cloud->resize (target_cloud->width * target_cloud->height);

        // plot the input and target
        for(int i = 0; i < target.cols(); i++)
        {
            target_cloud->points.at(i).x = target(0,i);
            target_cloud->points.at(i).y = target(1,i);
            target_cloud->points.at(i).z = target(2,i);
            // Draw the normal
            pcl::PointXYZ normal_vec{target_cloud->points.at(i).x + target_normals.at(i)(0),
                                     target_cloud->points.at(i).y + target_normals.at(i)(1),
                                     target_cloud->points.at(i).z + target_normals.at(i)(2)};      
            vis->addLine<pcl::PointXYZ> (target_cloud->points.at(i), normal_vec, std::string("normal")+std::to_string(i));            
        }

        vis->updatePointCloud(target_cloud, std::string("target"));

        vis->spinOnce(100000);        
    }
    
    Eigen::Matrix4f rigidTransform3D(Eigen::MatrixXf input, Eigen::MatrixXf target) override
    {
        std::cout << "Point to Plane" << std::endl; 
        
        Eigen::Matrix3f R;
        Eigen::Vector3f T;
        Eigen::Matrix4f Trans_total = Eigen::Matrix4f::Identity(); // accumulated homogenous transformation 

        std::vector<int16_t> correspondences(input.cols()); // same order as the input container : input -> target
        std::vector<int16_t> prev_correspond(correspondences); // same order as the input container : input -> target

        // 1) Find the normals of the target pointset
        std::vector<Eigen::Vector3f> target_normals(target.cols());
        findNormals(target, target_normals);
        std::cout << "size normals" << target_normals.size() << std::endl;

        for(int i = 0; i < max_iterations; i++)
        {
            // 2) Find the closest correspondences between input and target
            findCorrespondencesKnn(input, target, correspondences);

            // 3) Construct matrices for Ax=B, according to (10) in  
            // https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf
            Eigen::MatrixXf A(input.cols(), 6); // (Nx6)
            Eigen::VectorXf b(input.cols()); // (6x1)

            for(int j = 0; j < input.cols(); j++ )
            {
                float dx = target.col(correspondences.at(j))(0); 
                float dy = target.col(correspondences.at(j))(1);
                float dz = target.col(correspondences.at(j))(2);
                float nx = target_normals.at(correspondences.at(j))(0);
                float ny = target_normals.at(correspondences.at(j))(1);
                float nz = target_normals.at(correspondences.at(j))(2);

                float sx = input(0,j);
                float sy = input(1,j);
                float sz = input(2,j);

                A(j,0) = (nz * sy) - (ny * sz);
                A(j,1) = (nx * sz) - (nz * sx);
                A(j,2) = (ny * sx) - (nx * sy);
                A(j,3) = nx;
                A(j,4) = ny;
                A(j,5) = nz;

                b(j) = (nx * dx) + (ny * dy) + (nz * dz) - (nx * sx) - (ny * sy) - (nz * sz);
            }

            // 4) Solve linear system Ax=B, via SVD
            Eigen::VectorXf x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
            // Accumulate this iteration's transform into global transform
            Eigen::Matrix4f Trans = Eigen::Matrix4f::Identity();
            // X-Y-Z order
            R = Eigen::AngleAxisf(x(2), Eigen::Vector3f::UnitZ())
              * Eigen::AngleAxisf(x(1), Eigen::Vector3f::UnitY())
              * Eigen::AngleAxisf(x(0), Eigen::Vector3f::UnitX());          
            T = {x(3),x(4),x(5)};
            Trans.block<3,3>(0,0) = R;
            Trans.block<3,1>(0,3) = T;
            Trans_total =  Trans * Trans_total; // since R is left multiplier

            if(visualize) drawCorrespondences(input, target, correspondences);

            // 5) Apply the rotation to the last input
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

private:
    const int16_t N_NEIGHBORS_FOR_NORMALS;

};
} // namespace ICP