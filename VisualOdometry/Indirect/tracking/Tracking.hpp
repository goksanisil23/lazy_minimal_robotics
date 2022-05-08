#pragma once

#include <vector>
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include "matplotlibcpp.h"

namespace VISO
{

constexpr int32_t MAX_CORNERS = 3000;
constexpr int32_t KLT_WIN_SIZE = 21;
constexpr int32_t MIN_TRACKED_PTS = 100; // below this, we re-trigger feature detection
constexpr int32_t KLT_PYRAMIDS = 3;

class Tracking
{


public:
    Tracking(cv::Mat intrinsics_K) : K_(intrinsics_K), c_x_(K_.at<float>(0,2)), c_y_(K_.at<float>(1,2)), f_x_(K_.at<float>(0,0)), f_y_(K_.at<float>(1,1))
    {
        klt_termination_criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

        // matplotlibcpp::figure_size(600,400);     

        R_total = cv::Mat::eye(3,3, CV_64F);
        t_total = cv::Mat::zeros(3,1, CV_64F);  
    }

    void stepPnP(cv::Mat& rgb_image, cv::Mat& depth_image)
    {
        cv::Mat img_gray;
        cv::cvtColor(rgb_image, img_gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> keypoints;
        std::vector<cv::Point2f> tracked_keypoints;
        std::vector<cv::Point2f> good_tracked_keypoints;
        std::vector<cv::Point3d> model_points;
        std::vector<cv::Point2d> image_points;

        // 1) Track the keypoints from the previous frame with KLT optical flow
        // For verification, use the tracked points and execute reverse optical flow to see if the original keypoints match
        if (prev_keypoints_.size() > 0)
        {
            std::vector<uint8_t> status;
            std::vector<float> err;
            std::vector<cv::Point2f> backtracked_prev_keypoints;
            cv::calcOpticalFlowPyrLK(prev_img_, img_gray, prev_keypoints_, tracked_keypoints, status, err, cv::Size(KLT_WIN_SIZE,KLT_WIN_SIZE), 3, klt_termination_criteria_);
            cv::calcOpticalFlowPyrLK(img_gray, prev_img_, tracked_keypoints, backtracked_prev_keypoints, status, err, cv::Size(KLT_WIN_SIZE,KLT_WIN_SIZE), KLT_PYRAMIDS, klt_termination_criteria_);
            for(size_t i = 0; i < prev_keypoints_.size(); i++)
            {
                auto of_verif_dist = cv::norm(backtracked_prev_keypoints.at(i) - prev_keypoints_.at(i));
                if ( (status[i] == 1) && (of_verif_dist < 1.0) ) // if the flow for the corresponding feature is found
                {
                    good_tracked_keypoints.push_back(tracked_keypoints[i]);
                    
                    // Transform pixel coordinates to 3D camera coordinates using pinhole camera model
                    // 3D-camera points specified in (k-1), their matches specified in (k) in pixel coordinates
                    double depth = prev_depth_img_.at<double>(prev_keypoints_.at(i).y, prev_keypoints_.at(i).x);
                    double x_world = (prev_keypoints_.at(i).x - c_x_) * depth / f_x_;
                    double y_world = (prev_keypoints_.at(i).y - c_y_) * depth / f_y_;
                    cv::Point3d world_pt(x_world, y_world, depth); 
                    // cv::Point3d p_c = K_.inv() * depth *  homo_coord;
                    if(depth < 999.0)
                    {
                        model_points.push_back(world_pt); // 3d point at (k-1)
                        image_points.push_back( cv::Point2d(tracked_keypoints[i].x, tracked_keypoints[i].y) ); // corresponding image point at (k)
                    }
                    cv::line(rgb_image, tracked_keypoints[i], prev_keypoints_.at(i), cv::Scalar(0,0,255), 3);
                }
            }
            std::cout << "# good corners: " << good_tracked_keypoints.size() << std::endl;
            prev_keypoints_ = good_tracked_keypoints;
        }

        // 2) If tracked keypoints reduced below threshold, retrigger keypoint detection
        if( (prev_keypoints_.size() < MIN_TRACKED_PTS) ) 
        {
            cv::goodFeaturesToTrack(img_gray, prev_keypoints_, MAX_CORNERS, 0.3, 7, cv::Mat(), 7);
        }        

        
        if(model_points.size() > 5)
        {
            // 3) Estimate motion
            // Only use good matched keypoints, get the scale from the depth camera and use PnP
            cv::Mat R_rod;
            cv::solvePnPRansac(model_points, image_points, K_, cv::Mat::zeros(4,1,cv::DataType<double>::type), R_rod, t_current, false, 100, 1.0);
            cv::Rodrigues(R_rod, R_current);

            t_total = t_total + R_total * t_current;
            R_total = R_current * R_total;

            std::cout << "t_total:\n" << t_total << std::endl;
            
            x_traj.push_back(t_total.at<double>(0));
            y_traj.push_back(t_total.at<double>(1));
            z_traj.push_back(-t_total.at<double>(2));

            // // Visualize
            // matplotlibcpp::clf();
            matplotlibcpp::named_plot("viso",x_traj,z_traj,"-o");
            // matplotlibcpp::xlim(-40,100);
            // matplotlibcpp::ylim(-20,200);
            matplotlibcpp::grid(true);
            matplotlibcpp::legend();            
            matplotlibcpp::pause(0.0001);
        }


        ////////// Cosmetics ///////////
        for(auto kp : good_tracked_keypoints)
        {
            cv::circle(rgb_image, kp, 3, cv::Scalar(255,0,0), 2);
        }
        
        cv::imshow("tracked", rgb_image);
        cv::moveWindow("tracked", 30, 700);
        cv::waitKey(100);

        // // Update previous frame
        prev_img_ = img_gray.clone();
        prev_depth_img_ = depth_image.clone();
        frame_ctr++;

    }        

    cv::Vec3f rotationMatrixToEulerAngles(const cv::Mat &R)
    {

        float sy = std::sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

        bool singular = sy < 1e-6; // If

        float x, y, z;
        if (!singular)
        {
            x = std::atan2(R.at<double>(2,1) , R.at<double>(2,2));
            y = std::atan2(-R.at<double>(2,0), sy);
            z = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
        }
        else
        {
            x = std::atan2(-R.at<double>(1,2), R.at<double>(1,1));
            y = std::atan2(-R.at<double>(2,0), sy);
            z = 0;
        }
        return cv::Vec3f(x, y, z);

    }

    cv::Mat eulerAnglesToRotationMatrix(const cv::Vec3f &theta)
    {
        // Calculate rotation about x axis
        cv::Mat R_x = (cv::Mat_<double>(3,3) <<
                1,       0,              0,
                0,       std::cos(theta[0]),   -std::sin(theta[0]),
                0,       std::sin(theta[0]),   std::cos(theta[0])
                );

        // Calculate rotation about y axis
        cv::Mat R_y = (cv::Mat_<double>(3,3) <<
                std::cos(theta[1]),    0,      std::sin(theta[1]),
                0,               1,      0,
                -std::sin(theta[1]),   0,      std::cos(theta[1])
                );

        // Calculate rotation about z axis
        cv::Mat R_z = (cv::Mat_<double>(3,3) <<
                std::cos(theta[2]),    -std::sin(theta[2]),      0,
                std::sin(theta[2]),    std::cos(theta[2]),       0,
                0,               0,                  1);

        // Combined rotation matrix
        cv::Mat R = R_z * R_y * R_x;

        return R;

    }     

private:
    cv::Mat K_; // camera intrinsics
    float c_x_, c_y_, f_x_, f_y_;
    Eigen::Matrix4f T_; // current camera transformation matrix (Translation + Rotation)
    cv::TermCriteria klt_termination_criteria_;

    // Data associated to the previous image (for matching)
    cv::Mat prev_img_, prev_depth_img_;
    std::vector<cv::Point2f> prev_keypoints_;

    cv::Mat R_current;
    cv::Mat t_current;
    cv::Mat R_total;
    cv::Mat t_total;

    uint16_t frame_ctr = 0;

    std::vector<float> x_traj, y_traj, z_traj;

};


} // namespace VISO