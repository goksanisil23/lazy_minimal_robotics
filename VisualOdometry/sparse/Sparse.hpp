#pragma once

#include <vector>
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv4/opencv2/opencv.hpp>

namespace VISO
{

constexpr int32_t MAX_ORB_FEAUTURES = 500;
constexpr float LOWE_MATCH_RATIO = 0.5;

class Sparse
{

public:
    Sparse(cv::Mat K) : K_(K) 
    {
        orb_ = cv::ORB::create(MAX_ORB_FEAUTURES);
        flann_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    void step(cv::Mat& input_image)
    {
        // 1) Extract features & descriptors
        cv::Mat img_gray;
        cv::cvtColor(input_image, img_gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors; // (N_desriptor_size x N_keypoints_size)
        orb_->detectAndCompute(img_gray, cv::Mat(), keypoints, descriptors);
        descriptors.convertTo(descriptors, CV_32F);
        // cv::drawKeypoints(input_image, keypoints, input_image);

        // 2) Match against previous frame features
        std::vector<std::vector<cv::DMatch>> knn_matches;
        std::vector<cv::DMatch> good_matches;
        if( !descriptors.empty() && !prev_descriptors.empty())
        {
            flann_matcher_->knnMatch(descriptors, prev_descriptors, knn_matches, 2 ); // find 2 best matches, with distance in increasing order
            for(auto el : knn_matches)
            {
                // Take the better match, only if it's considerably more dominant than the next best match (considerably smaller distance)
                if(el[0].distance < LOWE_MATCH_RATIO * el[1].distance)
                    good_matches.push_back(el[0]);
            }
            cv::Mat img_matches;
            cv::drawMatches(img_gray, keypoints, prev_img_, prev_keypoints_, good_matches, img_matches, 
                cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            cv::imshow("matches", img_matches);
            cv::waitKey(10);
        }

        // 3) Estimate motion
        cv::Mat E, mask;
        E = cv::findEssentialMat(keypoints, prev_keypoints_, K(0,0), )



        // Update previous frame
        prev_img_ = std::move(img_gray);
        prev_keypoints_ = std::move(keypoints);
        prev_descriptors = std::move(descriptors);

    }

private:
    cv::Mat K_; // camera intrinsics
    Eigen::Matrix4f T_; // current camera transformation matrix (Translation + Rotation)
    cv::Ptr<cv::Feature2D> orb_;
    cv::Ptr<cv::DescriptorMatcher> flann_matcher_;


    // Data associated to the previous image (for matching)
    cv::Mat prev_img_;
    std::vector<cv::KeyPoint> prev_keypoints_;
    cv::Mat prev_descriptors;
};

} // namespace VISO