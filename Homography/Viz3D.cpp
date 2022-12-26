#include <thread>

#include "Viz3D.h"

namespace homography
{

Viz3D::Viz3D(const PlaneReference &planeRef, const cv::Matx33d &cameraIntrinsics)
    : cvViz3d_("pose from homography"), K_{cameraIntrinsics}
{
    // Show the world axes in 3D.
    cvViz3d_.showWidget("World-axes", cv::viz::WCoordinateSystem(0.2));

    cv::Mat refImgFlipped; // TODO: is rotation wrong or vis problem?
    cv::flip(planeRef.referenceImg_, refImgFlipped, 0);

    // Visualize the world plane as a 3D image.
    cvViz3d_.showWidget(
        "World-plane",
        cv::viz::WImage3D(refImgFlipped,
                          cv::Size2d{planeRef.planeReferenceObjWidth_, planeRef.planeReferenceObjHeight_},
                          cv::Vec3d::all(0.0),
                          {0.0, 0.0, -1.0},
                          {0.0, -1.0, 0.0}));

    cvViz3d_.setViewerPose(cv::Affine3d(cv::Matx33d{0.5001809878821235,
                                                    -0.01006515496432128,
                                                    -0.8658623863044331,
                                                    0.05308243649069191,
                                                    0.9984082550593005,
                                                    0.01905809973881113,
                                                    0.8642923315044942,
                                                    -0.05549458428519861,
                                                    0.4999191102743906},
                                        cv::Vec3d(1.79477, -0.182105, -0.52567)));
    cvViz3d_.setBackgroundColor();
}

void Viz3D::Render()
{
    while (true)
    {
        vizLock.lock();
        cvViz3d_.spinOnce();
        vizLock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
}

void Viz3D::Update(const cv::Mat &inputImg, const Eigen::Matrix4d &cameraPose)
{
    static int                       cameraCtr = 0;
    static std::vector<cv::Affine3d> cameraPosesCv;

    std::string cameraWidgetName{"Camera" + std::to_string(cameraCtr)};
    cv::Mat     cameraPoseCv;
    cv::eigen2cv(cameraPose, cameraPoseCv);
    cameraPosesCv.push_back(cv::Affine3d(cameraPoseCv));

    vizLock.lock();
    cvViz3d_.showWidget(cameraWidgetName, cv::viz::WCameraPosition(K_, inputImg, 0.5));
    cvViz3d_.setWidgetPose(cameraWidgetName, cv::Affine3d{cameraPoseCv});

    // Add the poses to the window as a trajectory
    // cv::viz::WTrajectory trajectory(cameraPosesCv, cv::viz::WTrajectory::PATH);
    // cvViz3d_.showWidget("Trajectory", trajectory);
    vizLock.unlock();
    // cameraCtr++;
}
} // namespace homography