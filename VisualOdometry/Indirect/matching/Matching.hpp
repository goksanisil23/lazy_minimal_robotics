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

    constexpr int32_t MAX_ORB_FEAUTURES = 3000;
    constexpr float LOWE_MATCH_RATIO = 0.5;

    class VO
    {

    public:
        VO(cv::Mat intrinsics_K) : K_(intrinsics_K), c_x_(K_.at<float>(0, 2)), c_y_(K_.at<float>(1, 2)), f_x_(K_.at<float>(0, 0)), f_y_(K_.at<float>(1, 1))
        {
            orb_ = cv::ORB::create(MAX_ORB_FEAUTURES);
            // orb_ =  cv::SIFT::create(MAX_ORB_FEAUTURES);
            flann_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            // flann_matcher_ = cv::FlannBasedMatcher()

            matplotlibcpp::figure_size(600, 400);

            R_total = cv::Mat::eye(3, 3, CV_64F);
            t_total = cv::Mat::zeros(3, 1, CV_64F);
        }

        void stepEssentialMatrixDecomp(cv::Mat &rgb_image, cv::Mat &depth_image)
        {
            // 1) Extract features & descriptors
            cv::Mat img_gray;
            cv::cvtColor(rgb_image, img_gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::KeyPoint> keypoints;
            std::vector<cv::Point2f> good_keypoints, good_prev_keypoints;
            cv::Mat descriptors, good_descriptors; // (N_desriptor_size x N_keypoints_size)
            orb_->detectAndCompute(img_gray, cv::Mat(), keypoints, descriptors);
            descriptors.convertTo(descriptors, CV_32F);
            // cv::drawKeypoints(input_image, keypoints, input_image);

            // 2) Match against previous frame features
            std::vector<std::vector<cv::DMatch>> knn_matches;
            std::vector<cv::DMatch> good_matches;
            if (!descriptors.empty() && !prev_descriptors.empty())
            {
                flann_matcher_->knnMatch(prev_descriptors, descriptors, knn_matches, 2); // find 2 best matches, with distance in increasing order
                for (auto el : knn_matches)
                {
                    // Take the better match, only if it's considerably more dominant than the next best match (considerably smaller distance)
                    if (el[0].distance < LOWE_MATCH_RATIO * el[1].distance)
                    {
                        good_matches.push_back(el[0]);
                        good_prev_keypoints.push_back(prev_keypoints_.at(el[0].queryIdx).pt);
                        good_keypoints.push_back(keypoints.at(el[0].trainIdx).pt);
                    }
                }
                cv::Mat img_matches;
                // only draw the good matches
                cv::drawMatches(prev_img_, prev_keypoints_, img_gray, keypoints, good_matches, img_matches,
                                cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cv::imshow("matches", img_matches);
                cv::moveWindow("matches", 30, 500);
                cv::waitKey(10);

                // 3) Estimate motion
                // Only use good matched keypoints to compute Essential matrix
                cv::Mat E, mask;
                E = cv::findEssentialMat(good_prev_keypoints, good_keypoints, K_.at<float>(0, 0), cv::Point2d(K_.at<float>(0, 2), K_.at<float>(1, 2)), cv::RANSAC, 0.999, 0.5);
                cv::recoverPose(E, good_prev_keypoints, good_keypoints, R_current, t_current, K_.at<float>(0, 0), cv::Point2d(K_.at<float>(0, 2), K_.at<float>(1, 2)));

                // if((abs_scale_> 0.1) ) // if sufficient camera movement
                {
                    // t_total = t_total +  (R_total* t_current);
                    t_total = t_total + abs_scale_ * (R_total * t_current);
                    cv::Vec3f angles = rotationMatrixToEulerAngles(R_total);
                    R_total = (R_current)*R_total;

                    x_traj.push_back(t_total.at<double>(0));
                    y_traj.push_back(t_total.at<double>(1));
                    z_traj.push_back(-t_total.at<double>(2));

                    // Visualize
                    matplotlibcpp::named_plot("viso", x_traj, z_traj, "-o");
                    matplotlibcpp::grid(true);
                    matplotlibcpp::legend();
                    matplotlibcpp::pause(0.0001);
                }
            }

            // Update previous frame
            prev_img_ = std::move(img_gray);
            prev_keypoints_ = std::move(keypoints);
            prev_descriptors = std::move(descriptors);
        }

        void stepPnP(cv::Mat &rgb_image, cv::Mat &depth_image)
        {
            // 1) Extract features & descriptors
            cv::Mat img_gray;
            cv::cvtColor(rgb_image, img_gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::KeyPoint> keypoints;
            std::vector<cv::Point2f> good_keypoints, good_prev_keypoints;
            std::vector<cv::Point3d> model_points;
            std::vector<cv::Point2d> image_points;
            cv::Mat descriptors, good_descriptors; // (N_desriptor_size x N_keypoints_size)
            orb_->detectAndCompute(img_gray, cv::Mat(), keypoints, descriptors);
            descriptors.convertTo(descriptors, CV_32F);

            // 2) Match against previous frame features
            std::vector<std::vector<cv::DMatch>> knn_matches;
            std::vector<cv::DMatch> good_matches;
            if (!descriptors.empty() && !prev_descriptors.empty())
            {
                flann_matcher_->knnMatch(prev_descriptors, descriptors, knn_matches, 2); // find 2 best matches, with distance in increasing order
                for (auto el : knn_matches)
                {
                    // Take the better match, only if it's considerably more dominant than the next best match (considerably smaller distance)
                    if (el[0].distance < LOWE_MATCH_RATIO * el[1].distance)
                    {
                        good_matches.push_back(el[0]);
                        good_prev_keypoints.push_back(prev_keypoints_.at(el[0].queryIdx).pt);
                        good_keypoints.push_back(keypoints.at(el[0].trainIdx).pt);

                        // Transform pixel coordinates to 3D camera coordinates using pinhole camera model
                        // 3D-camera points specified in (k-1), their matches specified in (k) in pixel coordinates
                        double depth = prev_depth_img_.at<double>(good_prev_keypoints.back().y, good_prev_keypoints.back().x);
                        double x_world = (good_prev_keypoints.back().x - c_x_) * depth / f_x_;
                        double y_world = (good_prev_keypoints.back().y - c_y_) * depth / f_y_;
                        cv::Point3d world_pt(x_world, y_world, depth);
                        // cv::Point3d p_c = K_.inv() * depth *  homo_coord;
                        if (depth < 999.0)
                        {
                            model_points.push_back(world_pt);                                                      // 3d point at (k-1)
                            image_points.push_back(cv::Point2d(good_keypoints.back().x, good_keypoints.back().y)); // corresponding image point at (k)
                        }
                    }
                }
                cv::Mat img_matches;
                // only draw the good matches
                cv::drawMatches(prev_img_, prev_keypoints_, img_gray, keypoints, good_matches, img_matches,
                                cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cv::imshow("matches", img_matches);
                cv::moveWindow("matches", 30, 500);
                cv::waitKey(100);

                // 3) Estimate motion
                // Only use good matched keypoints, get the scale from the depth camera and use PnP
                cv::Mat R_rod;
                cv::solvePnPRansac(model_points, image_points, K_, cv::Mat::zeros(4, 1, cv::DataType<double>::type), R_rod, t_current);
                cv::Rodrigues(R_rod, R_current);

                t_total = t_total + R_total * t_current;
                R_total = R_current * R_total;

                std::cout << "t_total:\n"
                          << t_total << std::endl;

                x_traj.push_back(t_total.at<double>(0));
                y_traj.push_back(t_total.at<double>(1));
                z_traj.push_back(-t_total.at<double>(2));

                // // Visualize
                // matplotlibcpp::clf();
                matplotlibcpp::named_plot("viso", x_traj, z_traj, "-o");
                matplotlibcpp::xlim(-10, 100);
                matplotlibcpp::ylim(-20, 150);
                matplotlibcpp::grid(true);
                matplotlibcpp::legend();
                matplotlibcpp::pause(0.0001);
            }

            // Update previous frame
            prev_img_ = img_gray.clone();
            prev_depth_img_ = depth_image.clone();
            prev_keypoints_ = std::move(keypoints);
            prev_descriptors = std::move(descriptors);
        }

        void updateAbsScale(const float &x_truth, const float &y_truth, const float &z_truth)
        {
            static float x_truth_prev, y_truth_prev, z_truth_prev;
            static float is_initial = true;
            if (is_initial)
            {
                x_truth_prev = x_truth;
                y_truth_prev = y_truth;
                z_truth_prev = z_truth;
                is_initial = false;
                abs_scale_ = 0.0;
            }
            else
            {
                abs_scale_ = std::sqrt((x_truth - x_truth_prev) * (x_truth - x_truth_prev) + (y_truth - y_truth_prev) * (y_truth - y_truth_prev) + (z_truth - z_truth_prev) * (z_truth - z_truth_prev));
                x_truth_prev = x_truth;
                y_truth_prev = y_truth;
                z_truth_prev = z_truth;
            }
        }

        cv::Vec3f rotationMatrixToEulerAngles(const cv::Mat &R)
        {

            float sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));

            bool singular = sy < 1e-6; // If

            float x, y, z;
            if (!singular)
            {
                x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
                y = std::atan2(-R.at<double>(2, 0), sy);
                z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
            }
            else
            {
                x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
                y = std::atan2(-R.at<double>(2, 0), sy);
                z = 0;
            }
            return cv::Vec3f(x, y, z);
        }

        cv::Mat eulerAnglesToRotationMatrix(const cv::Vec3f &theta)
        {
            // Calculate rotation about x axis
            cv::Mat R_x = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                           0, std::cos(theta[0]), -std::sin(theta[0]),
                           0, std::sin(theta[0]), std::cos(theta[0]));

            // Calculate rotation about y axis
            cv::Mat R_y = (cv::Mat_<double>(3, 3) << std::cos(theta[1]), 0, std::sin(theta[1]),
                           0, 1, 0,
                           -std::sin(theta[1]), 0, std::cos(theta[1]));

            // Calculate rotation about z axis
            cv::Mat R_z = (cv::Mat_<double>(3, 3) << std::cos(theta[2]), -std::sin(theta[2]), 0,
                           std::sin(theta[2]), std::cos(theta[2]), 0,
                           0, 0, 1);

            // Combined rotation matrix
            cv::Mat R = R_z * R_y * R_x;

            return R;
        }

    private:
        cv::Mat K_; // camera intrinsics
        float c_x_, c_y_, f_x_, f_y_;
        Eigen::Matrix4f T_; // current camera transformation matrix (Translation + Rotation)
        cv::Ptr<cv::Feature2D> orb_;
        cv::Ptr<cv::Feature2D> sift_;
        cv::Ptr<cv::DescriptorMatcher> flann_matcher_;

        // Data associated to the previous image (for matching)
        cv::Mat prev_img_, prev_depth_img_;
        std::vector<cv::KeyPoint> prev_keypoints_;
        cv::Mat prev_descriptors;

        cv::Mat R_current;
        cv::Mat t_current;
        cv::Mat R_total;
        cv::Mat t_total;

        std::vector<float> x_traj, y_traj, z_traj;
        bool isInitial = true;
        float abs_scale_;
    };

} // namespace VISO