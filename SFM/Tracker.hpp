#pragma once
#include <limits>

#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "matplotlibcpp.h"
#include "open3d/Open3D.h"

namespace sfm
{

    constexpr double DEPTH_THRESHOLD = 999.0;

    // Feature matching parameters
    constexpr int32_t MAX_ORB_FEAUTURES = 3000;
    constexpr float LOWE_MATCH_RATIO = 0.5;
    constexpr double MIN_MEAN_PIXEL_DISTANCE = 5.0;
    constexpr int MIN_MATCHES = 20;
    constexpr int NUM_PNP_ITERATIONS = 100;
    constexpr float PNP_REPROJECTION_ERR_THRES = 4.0F;
    constexpr double PNP_CONFIDENCE = 0.99;

    // Feature tracking parameters
    constexpr int32_t MAX_CORNERS = 3000;
    constexpr int32_t KLT_WIN_SIZE = 21;
    // below this, we re-trigger feature detection
    constexpr int32_t MIN_TRACKED_PTS = 100;
    constexpr int32_t KLT_PYRAMIDS = 3;

    const std::string RGB_PCD_PATH_PREFIX = "../resources/data/pcds/after_viso/rgb/rgb_cloud_";
    const std::string KP_PCD_PATH_PREFIX = "../resources/data/pcds/after_viso/kp/keypoint_cloud_";

    struct Landmark
    {
        uint32_t idx;
        std::vector<uint32_t> camera_indices;
        std::vector<cv::Point2d> pixel_coords;
        Eigen::Vector3d translation; // w.r.t initial camera
        Eigen::Matrix3d rotation;    // w.r.t initial camera
    };

    struct MatchedFeatureCoords
    {
        MatchedFeatureCoords(const float &xk_minus, const float &yk_minus, const float &xk, const float &yk)
            : x_k_minus{xk_minus}, y_k_minus{yk_minus}, x_k{xk}, y_k{yk} {}

        // x,y pixel coordinates of the matched feature at frame k-1
        float x_k_minus, y_k_minus;
        // x,y pixel coordinate of the matched feature at frame k
        float x_k, y_k;
    };

    class Tracker
    {
    public:
        Tracker()
        {
            // ------------ OAK-D ------------ //
            // _cX = 1917.1802978515625;
            // _cY = 1075.5216064453125;
            // _fX = 3094.48486328125;
            // _fY = 3094.48486328125;

            // const double imageHeightOriginal = 2160;
            // const double imageWidthOriginal = 3840;

            // // const double imageHeight = 120;
            // // const double imageWidth = 160;
            // const double imageHeight = 360;
            // const double imageWidth = 640;

            // const double aspRatioX = imageWidthOriginal / imageHeight;
            // const double aspRatioY = imageWidthOriginal / imageWidth;

            // _cX = _cX / aspRatioX;
            // _cY = _cY / aspRatioY;
            // _fX = _fX / aspRatioX;
            // _fY = _fY / aspRatioY;

            // ------------------ CARLA ---------- //
            _cX = 512.0f;
            _cY = 320.0f;
            _fX = 512.0f;
            _fY = 512.0f;

            // ------------------------------------- //

            K_ = (cv::Mat_<float>(3, 3) << _fX, 0.0, _cX, 0.0, _fY, _cY, 0.0, 0.0, 1.0);

            orb_ = cv::ORB::create(MAX_ORB_FEAUTURES);
            flann_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

            klt_termination_criteria_ = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

            R_total = cv::Mat::eye(3, 3, CV_64F);
            t_total = cv::Mat::zeros(3, 1, CV_64F);
            R_prev = cv::Mat::eye(3, 3, CV_64F);
            t_prev = cv::Mat::zeros(3, 1, CV_64F);

            // Initialize pointcloud visualizer
            // setup visualization
            // vis.CreateVisualizerWindow("projected camera", 960, 540, 480, 270);
            // vis.GetRenderOption().background_color_ = {0.05, 0.05, 0.05};
            // vis.GetRenderOption().point_size_ = 3;
            // vis.GetRenderOption().show_coordinate_frame_ = true;

            matplotlibcpp::figure_size(600, 400);
            data_for_ba_txt_.open("../resources/data_for_ba.txt");
            data_for_ba_txt_ << "image_k_minus feature_k_minus_x_coord feature_k_minus_y_coord "
                                "image_k feature_k_x_coord feature_k_y_coord "
                                "3d_world_pt_x 3d_world_pt_y 3d_world_pt_z"
                             << std::endl;
        }

        void inline transformPoint(const Eigen::Matrix3d &rot, const Eigen::Vector3d &trans, Eigen::Vector3d &pt)
        {
            auto t1 = pt(0);
            auto t2 = pt(1);
            auto t3 = pt(2);

            pt(0) = rot(0, 0) * t1 + rot(0, 1) * t2 + rot(0, 2) * t3 + trans(0);
            pt(1) = rot(1, 0) * t1 + rot(1, 1) * t2 + rot(1, 2) * t3 + trans(1);
            pt(2) = rot(2, 0) * t1 + rot(2, 1) * t2 + rot(2, 2) * t3 + trans(2);
        }

        void transformPointsTo3dLandmarks(const int &imgIdx,
                                          const std::vector<cv::Point3d> &model_points,
                                          const Eigen::Matrix3d &rot, const Eigen::Vector3d &trans,
                                          std::vector<Eigen::Vector3d> &landmark_points)
        {
            for (auto pt_in_cam_frame : model_points)
            {
                Eigen::Vector3d pt_in_world_frame = Eigen::Vector3d(-pt_in_cam_frame.x,
                                                                    -pt_in_cam_frame.y,
                                                                    pt_in_cam_frame.z);
                transformPoint(rot, trans, pt_in_world_frame);
                landmark_points.push_back(pt_in_world_frame);
            }
            // Write to file
            o3d_cloud = std::make_shared<open3d::geometry::PointCloud>(landmark_points);
            o3d_cloud->points_ = landmark_points;
            // o3d_cloud->colors_ = o3d_colors;
            open3d::io::WritePointCloud(KP_PCD_PATH_PREFIX + std::to_string(imgIdx) + ".pcd", *o3d_cloud);
        }

        void projectImageTo3d(const int &imgIdx,
                              const cv::Mat &rgbImage, const cv::Mat &depthImage,
                              const Eigen::Matrix3d &rot = Eigen::Matrix3d::Identity(),
                              const Eigen::Vector3d &trans = Eigen::Vector3d::Zero())
        {
            std::vector<Eigen::Vector3d> o3d_points;
            std::vector<Eigen::Vector3d> o3d_colors;

            for (int ii = 0; ii < rgbImage.rows; ii++)
            {
                for (int jj = 0; jj < rgbImage.cols; jj++)
                {
                    double depth = depthImage.at<double>(ii, jj);
                    if ((depth > 0) && (depth < DEPTH_THRESHOLD)) // its possible that stereo depth returns 0
                    {
                        double x_world = (jj - _cX) * depth / _fX;
                        double y_world = (ii - _cY) * depth / _fY;
                        double z_world = depth;
                        Eigen::Vector3d pt_in_cam_frame(-x_world, -y_world, z_world);
                        Eigen::Vector3d pt_in_world_frame = pt_in_cam_frame;
                        transformPoint(rot, trans, pt_in_world_frame);

                        // o3d_points.at(jj + ii * rgbImage.cols) = pt_in_cam_frame;
                        o3d_points.push_back(pt_in_world_frame);
                        auto rgbColor = rgbImage.at<cv::Vec3b>(ii, jj);
                        o3d_colors.push_back(Eigen::Vector3d(rgbColor[2], rgbColor[1], rgbColor[0]) / 255.0);
                    }
                }
            }
            // Write to file
            o3d_cloud = std::make_shared<open3d::geometry::PointCloud>(o3d_points);
            o3d_cloud->points_ = o3d_points;
            o3d_cloud->colors_ = o3d_colors;
            open3d::io::WritePointCloud(RGB_PCD_PATH_PREFIX + std::to_string(imgIdx) + ".pcd", *o3d_cloud);
        }

        void stepPnp(cv::Mat &rgb_image, cv::Mat &depth_image)
        {
            static int imgIdx = 0;
            // 1) Extract features & descriptors
            cv::Mat img_gray;
            cv::cvtColor(rgb_image, img_gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::KeyPoint> keypoints;
            std::vector<cv::Point2f> good_keypoints, good_prev_keypoints;
            std::vector<cv::Point3d> model_points;
            std::vector<cv::Point2d> image_points;
            std::vector<MatchedFeatureCoords> matched_feature_coords;
            cv::Mat descriptors, good_descriptors; // (N_desriptor_size x N_keypoints_size)
            orb_->detectAndCompute(img_gray, cv::Mat(), keypoints, descriptors);
            descriptors.convertTo(descriptors, CV_32F);

            // 2) Match against previous frame features
            std::vector<std::vector<cv::DMatch>> knn_matches;
            std::vector<cv::DMatch> good_matches;
            double meanPixelDistance = 0.0;

            if (!descriptors.empty() && !prev_descriptors_.empty())
            {
                // find 2 best matches, with distance in increasing order
                flann_matcher_->knnMatch(prev_descriptors_, descriptors, knn_matches, 2);
                for (auto el : knn_matches)
                {
                    // Take the better match, only if it's considerably more dominant than
                    // the next best match (considerably smaller distance)
                    if (el[0].distance < LOWE_MATCH_RATIO * el[1].distance)
                    {
                        good_matches.push_back(el[0]);
                        good_prev_keypoints.push_back(prev_keypoints_.at(el[0].queryIdx).pt);
                        good_keypoints.push_back(keypoints.at(el[0].trainIdx).pt);

                        meanPixelDistance += calculateMeanPixelDist(
                            good_prev_keypoints.back(), good_keypoints.back());

                        // Transform pixel coordinates to 3D camera coordinates using pinhole camera model.
                        // 3D-camera points specified in (k-1), their matches specified in (k) in pixel coordinates
                        double depth = prev_depth_img_.at<double>(good_prev_keypoints.back().y,
                                                                  good_prev_keypoints.back().x);
                        if ((depth > 0) && (depth < DEPTH_THRESHOLD))
                        {
                            double x_world = (good_prev_keypoints.back().x - _cX) * depth / _fX;
                            double y_world = (good_prev_keypoints.back().y - _cY) * depth / _fY;
                            cv::Point3d world_pt(x_world, y_world, depth);
                            // 3d point at (k-1)
                            model_points.push_back(world_pt);
                            // corresponding image point at (k)
                            image_points.push_back(cv::Point2d(good_keypoints.back().x, good_keypoints.back().y));

                            auto mf_coords = MatchedFeatureCoords(good_prev_keypoints.back().x, good_prev_keypoints.back().y,
                                                                  good_keypoints.back().x, good_keypoints.back().y);
                            matched_feature_coords.push_back(mf_coords);
                        }
                    }
                }
                meanPixelDistance = meanPixelDistance / good_prev_keypoints.size();

                if ((image_points.size() > MIN_MATCHES))
                // if ((image_points.size() > MIN_MATCHES) && (meanPixelDistance > MIN_MEAN_PIXEL_DISTANCE))
                {
                    cv::Mat img_matches;
                    // only draw the good matches
                    cv::drawMatches(prev_img_, prev_keypoints_, img_gray, keypoints,
                                    good_matches, img_matches, cv::Scalar::all(-1),
                                    cv::Scalar::all(-1), std::vector<char>(),
                                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                    cv::imshow("matches", img_matches);
                    cv::moveWindow("matches", 30, 500);
                    cv::waitKey(10);

                    // 3) Estimate motion
                    // Only use good matched keypoints, get the scale from the depth camera
                    // and use PnP
                    cv::Mat R_rod;
                    // Reprojection error parameter can really make a difference between getting good/bad results.
                    // Default value of 8.0 was giving a jumpy pose yet 4.0 works fine
                    std::vector<int> inlier_idxs;
                    cv::solvePnPRansac(model_points, image_points, K_, cv::Mat::zeros(4, 1, cv::DataType<double>::type),
                                       R_rod, t_current, false,
                                       NUM_PNP_ITERATIONS, PNP_REPROJECTION_ERR_THRES, PNP_CONFIDENCE,
                                       inlier_idxs);
                    cv::Rodrigues(R_rod, R_current);
                    std::cout << "all: " << image_points.size() << " inliers: " << inlier_idxs.size() << std::endl;

                    t_prev = t_total;
                    R_prev = R_total;
                    t_total = t_prev + R_prev * t_current;
                    R_total = R_current * R_prev;

                    // Project to 3d and translate to world coordinates
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        eigenR(R_total.ptr<double>(), R_total.rows, R_total.cols);
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        eigenT(t_total.ptr<double>(), t_total.rows, t_total.cols);
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        eigenR_prev(R_prev.ptr<double>(), R_prev.rows, R_prev.cols);
                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                        eigenT_prev(t_prev.ptr<double>(), t_prev.rows, t_prev.cols);
                    projectImageTo3d(imgIdx, rgb_image, depth_image, eigenR, eigenT);

                    x_traj.push_back(t_total.at<double>(0));
                    y_traj.push_back(t_total.at<double>(1));
                    z_traj.push_back(t_total.at<double>(2));

                    std::vector<Eigen::Vector3d> landmark_points;
                    // use transform of idx k-1 since the 3d model points (model_points) belong to k-1
                    transformPointsTo3dLandmarks(imgIdx - 1, model_points, eigenR_prev, eigenT_prev, landmark_points);
                    saveForBA(imgIdx, matched_feature_coords, inlier_idxs, landmark_points);

                    // // Visualize
                    matplotlibcpp::clf();
                    matplotlibcpp::named_plot("viso", z_traj, x_traj, "-o");
                    matplotlibcpp::grid(true);
                    matplotlibcpp::legend();
                    matplotlibcpp::pause(0.0001);
                }
                else
                {
                    // If current frame did not provide enough matches with the previous
                    // frame, we should not use it. To ensure continous motion, instead of
                    // ignoring the previous frame, we ignore the current frame.
                    return;
                }
            }

            // Update previous frame
            imgIdx++;
            prev_img_ = img_gray.clone();
            prev_depth_img_ = depth_image.clone();
            prev_keypoints_ = std::move(keypoints);
            prev_descriptors_ = std::move(descriptors);
        }

        void saveForBA(const int &imgIdx, const std::vector<MatchedFeatureCoords> &matched_feature_coords,
                       const std::vector<int> &inlier_idxs, const std::vector<Eigen::Vector3d> &landmark_points)
        {
            // matched_feature_coords & landmark_points are aligned in size and indices
            for (auto idx : inlier_idxs)
            {
                // Round the pixel coordinates in order to allow matching later on
                data_for_ba_txt_ << imgIdx - 1
                                 << " " << (matched_feature_coords.at(idx).x_k_minus)
                                 << " " << (matched_feature_coords.at(idx).y_k_minus)
                                 << " " << imgIdx
                                 << " " << (matched_feature_coords.at(idx).x_k)
                                 << " " << (matched_feature_coords.at(idx).y_k_minus)
                                 << " " << landmark_points.at(idx)(0) << " " << landmark_points.at(idx)(1) << " " << landmark_points.at(idx)(2)
                                 << std::endl;
            }
        }

        double calculateMeanPixelDist(const cv::Point2f &pt1,
                                      const cv::Point2f &pt2)
        {
            double dx = pt1.x - pt2.x;
            double dy = pt1.y - pt2.y;
            return sqrt(dx * dx + dy * dy);
        }

        void stepPnpKlt(cv::Mat &rgb_image, cv::Mat &depth_image)
        {
            cv::Mat img_gray;
            cv::cvtColor(rgb_image, img_gray, cv::COLOR_BGR2GRAY);
            std::vector<cv::Point2f> keypoints;
            std::vector<cv::Point2f> tracked_keypoints;
            std::vector<cv::Point2f> good_tracked_keypoints;
            std::vector<cv::Point3d> model_points;
            std::vector<cv::Point2d> image_points;

            // 1) Track the keypoints from the previous frame with KLT optical flow
            // For verification, use the tracked points and execute reverse optical flow
            // to see if the original keypoints match
            if (prev_keypoints_klt_.size() > 0)
            {
                std::vector<uint8_t> status;
                std::vector<float> err;
                std::vector<cv::Point2f> backtracked_prev_keypoints;
                cv::calcOpticalFlowPyrLK(
                    prev_img_, img_gray, prev_keypoints_klt_, tracked_keypoints, status, err,
                    cv::Size(KLT_WIN_SIZE, KLT_WIN_SIZE), 3, klt_termination_criteria_);
                cv::calcOpticalFlowPyrLK(img_gray, prev_img_, tracked_keypoints,
                                         backtracked_prev_keypoints, status, err,
                                         cv::Size(KLT_WIN_SIZE, KLT_WIN_SIZE),
                                         KLT_PYRAMIDS, klt_termination_criteria_);
                for (size_t i = 0; i < prev_keypoints_klt_.size(); i++)
                {
                    auto of_verif_dist =
                        cv::norm(backtracked_prev_keypoints.at(i) - prev_keypoints_klt_.at(i));
                    if ((status[i] == 1) &&
                        (of_verif_dist <
                         1.0)) // if the flow for the corresponding feature is found
                    {
                        good_tracked_keypoints.push_back(tracked_keypoints[i]);

                        // Transform pixel coordinates to 3D camera coordinates using pinhole
                        // camera model 3D-camera points specified in (k-1), their matches
                        // specified in (k) in pixel coordinates
                        double depth = prev_depth_img_.at<double>(prev_keypoints_klt_.at(i).y,
                                                                  prev_keypoints_klt_.at(i).x);
                        if (depth > 0)
                        {
                            double x_world = (prev_keypoints_klt_.at(i).x - _cX) * depth / _fX;
                            double y_world = (prev_keypoints_klt_.at(i).y - _cY) * depth / _fY;
                            cv::Point3d world_pt(x_world, y_world, depth);
                            model_points.push_back(world_pt); // 3d point at (k-1)
                            image_points.push_back(cv::Point2d(
                                tracked_keypoints[i].x,
                                tracked_keypoints[i].y)); // corresponding image point at (k)
                            cv::line(rgb_image, tracked_keypoints[i], prev_keypoints_klt_.at(i),
                                     cv::Scalar(0, 0, 255), 3);
                        }
                    }
                }
                std::cout << "# good corners: " << good_tracked_keypoints.size()
                          << std::endl;
                prev_keypoints_klt_ = good_tracked_keypoints;
            }

            // 2) If tracked keypoints reduced below threshold, retrigger keypoint
            // detection
            if ((prev_keypoints_klt_.size() < MIN_TRACKED_PTS))
            {
                cv::goodFeaturesToTrack(img_gray, prev_keypoints_klt_, MAX_CORNERS, 0.3, 7,
                                        cv::Mat(), 7);
            }

            if (model_points.size() > MIN_MATCHES)
            {
                // 3) Estimate motion
                // Only use good matched keypoints, get the scale from the depth camera
                // and use PnP
                cv::Mat R_rod;
                cv::solvePnPRansac(model_points, image_points, K_,
                                   cv::Mat::zeros(4, 1, cv::DataType<double>::type),
                                   R_rod, t_current, false, 100, 1.0);
                cv::Rodrigues(R_rod, R_current);

                t_total = t_total + R_total * t_current;
                R_total = R_current * R_total;

                std::cout << "t_total:\n"
                          << t_total << std::endl;

                x_traj.push_back(t_total.at<double>(0));
                y_traj.push_back(t_total.at<double>(1));
                z_traj.push_back(-t_total.at<double>(2));

                // // Visualize
                matplotlibcpp::clf();
                matplotlibcpp::named_plot("viso", x_traj, z_traj, "-o");
                // matplotlibcpp::xlim(-40,100);
                // matplotlibcpp::ylim(-20,200);
                matplotlibcpp::grid(true);
                matplotlibcpp::legend();
                matplotlibcpp::pause(0.0001);
            }

            ////////// Cosmetics ///////////
            for (auto kp : good_tracked_keypoints)
            {
                cv::circle(rgb_image, kp, 3, cv::Scalar(255, 0, 0), 2);
            }

            // cv::imshow("tracked", rgb_image);
            // cv::moveWindow("tracked", 30, 700);
            // cv::waitKey(10);

            // // Update previous frame
            prev_img_ = img_gray.clone();
            prev_depth_img_ = depth_image.clone();
        }

        ~Tracker()
        {
            data_for_ba_txt_.close();
        }

    private:
        // camera intrinsics
        double _cX, _cY, _fX, _fY;
        cv::Mat K_;

        open3d::visualization::Visualizer vis;
        std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud;

        // current camera transformation matrix (Translation + Rotation)
        Eigen::Matrix4f T_;
        cv::Ptr<cv::Feature2D> orb_;
        cv::Ptr<cv::Feature2D> sift_;
        cv::Ptr<cv::DescriptorMatcher> flann_matcher_;

        // Data associated to the previous image (for matching)
        cv::Mat prev_img_, prev_depth_img_;
        std::vector<cv::KeyPoint> prev_keypoints_;
        std::vector<cv::Point2f> prev_keypoints_klt_;
        cv::Mat prev_descriptors_;

        cv::TermCriteria klt_termination_criteria_;

        cv::Mat R_current, t_current;
        cv::Mat R_total, t_total;
        cv::Mat R_prev, t_prev;

        std::vector<float> x_traj, y_traj, z_traj;
        std::ofstream data_for_ba_txt_;
    };
} // namespace sfm
