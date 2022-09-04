
#include <iostream>
#include <memory>
#include <filesystem>

#include "open3d/Open3D.h"
#include "matplotlibcpp.h"

int main(int argc, char *argv[])
{
    // matplotlibcpp::figure_size(600, 400);
    std::vector<double> x_traj, y_traj, z_traj;
    x_traj.push_back(0);
    y_traj.push_back(0);
    z_traj.push_back(0);

    // setup visualization
    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("projected camera", 960, 540, 480, 270);

    using namespace open3d;

    const std::string BASE_PATH = "/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/rgbd-pointcloud/imgs2/";
    // std::string intrinsic_path(BASE_PATH + std::string("intrinsics_o3d.json"));
    std::string intrinsic_path("/home/goksan/Downloads/depthai-experiments/gen2-pointcloud/rgbd-pointcloud/imgs3/intrinsics_o3d.json");
    camera::PinholeCameraIntrinsic intrinsic;
    io::ReadIJsonConvertible(intrinsic_path, intrinsic);

    //   Read files
    std::string depthFilesDir(BASE_PATH + std::string("depths/"));
    std::string rgbFilesDir(BASE_PATH + std::string("jpegs/"));
    std::vector<std::string> _depthImageFiles, _rgbImageFiles;
    std::vector<std::string>::iterator _depthImgsItr, _rgbImgsItr;

    for (const auto &file :
         std::filesystem::directory_iterator(depthFilesDir))
    {
        if (file.path().string().find("depth_") != std::string::npos)
        {
            _depthImageFiles.emplace_back(file.path().string());
        }
    }
    for (const auto &file : std::filesystem::directory_iterator(rgbFilesDir))
    {
        if (file.path().string().find("rgb_") != std::string::npos)
        {
            _rgbImageFiles.emplace_back(file.path().string());
        }
    }
    // Sort the files alphabetically (chronologically)
    std::sort(_depthImageFiles.begin(), _depthImageFiles.end());
    std::sort(_rgbImageFiles.begin(), _rgbImageFiles.end());
    _depthImgsItr = _depthImageFiles.begin();
    _rgbImgsItr = _rgbImageFiles.begin();

    Eigen::Matrix4d T_total = Eigen::Matrix4d::Identity();

    while (_depthImgsItr != (_depthImageFiles.end() - 1))
    {
        auto color_source = io::CreateImageFromFile(*_rgbImgsItr);
        auto depth_source = io::CreateImageFromFile(*_depthImgsItr);
        auto color_target = io::CreateImageFromFile(*(_rgbImgsItr + 1));
        auto depth_target = io::CreateImageFromFile(*(_depthImgsItr + 1));
        std::shared_ptr<geometry::RGBDImage> (*CreateRGBDImage)(
            const geometry::Image &, const geometry::Image &, bool);
        CreateRGBDImage = &geometry::RGBDImage::CreateFromRedwoodFormat;
        auto source = CreateRGBDImage(*color_source, *depth_source, true);
        auto target = CreateRGBDImage(*color_target, *depth_target, true);

        pipelines::odometry::OdometryOption option;
        Eigen::Matrix4d odo_init = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d trans_odo = Eigen::Matrix4d::Identity();
        Eigen::Matrix6d info_odo = Eigen::Matrix6d::Zero();
        bool is_success;

        {
            pipelines::odometry::RGBDOdometryJacobianFromColorTerm jacobian_method;
            std::tie(is_success, trans_odo, info_odo) =
                pipelines::odometry::ComputeRGBDOdometry(
                    *source, *target, intrinsic, odo_init, jacobian_method, option);
        }
        std::cout << "Estimated 4x4 motion matrix : " << std::endl;
        std::cout << trans_odo << std::endl;
        std::cout << "Estimated 6x6 information matrix : " << std::endl;
        std::cout << info_odo << std::endl;

        T_total = T_total * trans_odo;

        _rgbImgsItr = _rgbImgsItr + 10;
        _depthImgsItr = _depthImgsItr + 10;

        // // Visualize
        // x_traj.push_back(T_total(0, 3));
        // y_traj.push_back(T_total(1, 3));
        // z_traj.push_back(T_total(2, 3));

        // // // Visualize
        // matplotlibcpp::clf();
        // matplotlibcpp::named_plot("viso", x_traj, y_traj, "-o");
        // matplotlibcpp::grid(true);
        // matplotlibcpp::legend();
        // matplotlibcpp::pause(0.0001);

        auto mesh = geometry::TriangleMesh::CreateCoordinateFrame();
        mesh->Transform(T_total);
        vis.AddGeometry(mesh);
        vis.PollEvents();
        vis.UpdateRender();
    }
}
