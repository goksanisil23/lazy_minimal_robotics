#include <thread>

#include "Homography.h"
#include "Viz3D.h"

#include "oakd_image_reader.hpp"

const std::string OAKD_LEFT_CAM_IMAGES_DIR  = "../../Tools/OakDImageHandler/imgs";
const std::string TEMPLATE_PLANE_IMAGE_PATH = "/home/goksan/Downloads/toronto_template.jpg";

constexpr double REFERENCE_PLANE_OBJ_WIDTH  = 0.245; // [m]
constexpr double REFERENCE_PLANE_OBJ_HEIGHT = 0.085; // [m]

int main()
{
    // Set of images that we're going to match against the template plane
    oakd::OakdImageReader oakdImageReader(OAKD_LEFT_CAM_IMAGES_DIR.c_str());

    // Create an homography object that will do template plane matching based camera localization
    cv::Mat                templateImage{cv::imread(TEMPLATE_PLANE_IMAGE_PATH)};
    homography::Homography homography(homography::CAMERA_TYPE::OAK_D_RIGHT);
    homography.SetPlaneReferenceImage(templateImage, REFERENCE_PLANE_OBJ_WIDTH, REFERENCE_PLANE_OBJ_HEIGHT);

    // Visualizer
    homography::Viz3D visualizer(homography.GetPlaneRefCopy(), homography.GetCameraInstrinsics());
    std::thread       visThread(&homography::Viz3D::Render, &visualizer);

    int img_ctr = 0;

    cv::Mat oakdGrayImage;
    while (oakdImageReader.GetNextImage(oakdGrayImage))
    {
        // cv::Mat undistorted{homography.UndistortImage(oakdGrayImage)};

        homography.FindMatches(oakdGrayImage, templateImage);
        cv::Mat homographyMatrix = homography.ComputeHomography(homography::HOMOGRAPHY_MODE::IMAGE_TO_IMAGE);
        if (!homographyMatrix.empty())
            homography.LocateTemplatePlane(templateImage, oakdGrayImage, homographyMatrix);

        cv::Mat         homographyMatrix3d = homography.ComputeHomography(homography::HOMOGRAPHY_MODE::OBJ_TO_IMAGE);
        Eigen::Matrix4d cameraPose;
        if (!homographyMatrix3d.empty())
        {
            cameraPose = homography.ComputeCameraPose(homographyMatrix3d);
            // Visualize
            visualizer.Update(oakdGrayImage, cameraPose);
            if (img_ctr == 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        }

        cv::waitKey(1000);
        img_ctr++;
    }

    std::cout << "Finished\n";
    visThread.join();

    return 0;
}