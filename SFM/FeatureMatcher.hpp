#pragma once
#include <limits>
#include <optional>

#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/opencv.hpp>

#include "open3d/Open3D.h"

#include <matplot/matplot.h>

using namespace std::chrono_literals;

namespace sfm
{

constexpr double DEPTH_THRESHOLD = 999.0;

// Feature matching parameters
constexpr int32_t MAX_ORB_FEAUTURES          = 3000;
constexpr float   LOWE_MATCH_RATIO           = 0.5;
constexpr double  MIN_MEAN_PIXEL_DISTANCE    = 5.0;
constexpr int     MIN_MATCHES                = 20;
constexpr int     NUM_PNP_ITERATIONS         = 100;
constexpr float   PNP_REPROJECTION_ERR_THRES = 4.0F;
constexpr double  PNP_CONFIDENCE             = 0.99;

// Feature tracking parameters
constexpr int32_t MAX_CORNERS                          = 3000;
constexpr int     NUM_REQUIRED_VIEWPOINTS_FOR_LANDMARK = 3;

// drawing locations for 1024x640 images in a 3440x1440 screen
const std::vector<std::pair<int, int>> windowLocations{std::make_pair<int, int>(30, 30),
                                                       std::make_pair<int, int>(1030, 30),
                                                       std::make_pair<int, int>(2030, 30),
                                                       std::make_pair<int, int>(30, 700),
                                                       std::make_pair<int, int>(1030, 700),
                                                       std::make_pair<int, int>(2030, 700)};

const std::string RGB_PCD_PATH_PREFIX = "../resources/data/pcds/after_matching/rgb_cloud_";
const std::string KP_PCD_PATH_PREFIX  = "../resources/data/pcds/after_matching/keypoint_cloud_";
const std::string BA_DATA_PATH        = "../resources/data/data_for_ba.txt";

class FeatureMatcher
{
  public:
    // Data types
    struct DescriptorIndentifier
    {
        size_t imgIdx;  // image index descriptor belongs to
        int    descIdx; // descriptor index within this image index

        DescriptorIndentifier(const size_t &imgIdx_in, const int &descIdx_in) : imgIdx{imgIdx_in}, descIdx{descIdx_in}
        {
        }
    };

    struct RoboticsPose
    {
        RoboticsPose(const float &xIn,
                     const float &yIn,
                     const float &zIn,
                     const float &qxIn,
                     const float &qyIn,
                     const float &qzIn,
                     const float &qwIn)
            : x{xIn}, y{yIn}, z{zIn}, qx{qxIn}, qy{qyIn}, qz{qzIn}, qw{qwIn}
        {
        }

        float x, y, z;
        float qx, qy, qz, qw;
    };

    FeatureMatcher()
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

        orb_           = cv::ORB::create(MAX_ORB_FEAUTURES);
        flann_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

        baDataOutStream_.open(BA_DATA_PATH);
        // baDataOutStream_ << "num_cameras num_3d_landmarks num_observations" << std::endl;
        // baDataOutStream_ << "0 0 0" << std::endl; // to be replaced
    }

    void AddRgbDepthPair(const cv::Mat &rgbImg, const cv::Mat &depthImg)
    {
        rgbImgs_.push_back(rgbImg);
        depthImgs_.push_back(depthImg);

        keypointsAllImgs_.resize(rgbImgs_.size());
        descriptorsAllImgs_.resize(rgbImgs_.size());
    }

    void AddCameraPose(const RoboticsPose &cameraPose)
    {
        cameraPoses_.push_back(cameraPose);
    }

    void FindAllKeypointsAndDescriptors()
    {
        for (size_t imgIdx = 0; imgIdx < rgbImgs_.size(); imgIdx++)
        {
            cv::Mat imgGray;
            cv::cvtColor(rgbImgs_.at(imgIdx), imgGray, cv::COLOR_BGR2GRAY);
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat                   descriptors; // (num_keypoints x descriptor_size) = (3000 x 32)
            orb_->detectAndCompute(imgGray, cv::Mat(), keypoints, descriptors);
            keypointsAllImgs_.at(imgIdx) = keypoints;
            descriptors.convertTo(descriptors, CV_32F); // descriptors are integers but for knn we neet float
            descriptorsAllImgs_.at(imgIdx) = descriptors;
        }

        // Allocate size for (num_imgs) x (num_imgs) x (num_matches_for_i_j_image_pair) x (source_descriptor_index, target_descriptor_index)
        globalMatchInfo_.resize(rgbImgs_.size());
        for (auto &el : globalMatchInfo_)
        {
            el.resize(rgbImgs_.size());
        }
    }

    std::vector<char> maskMatchesByTrainImgIdx(const std::vector<cv::DMatch> &matches, int trainImgIdx)
    {
        std::vector<char> mask;
        mask.resize(matches.size());
        std::fill(mask.begin(), mask.end(), 0);
        for (size_t i = 0; i < matches.size(); i++)
        {
            if (matches.at(i).imgIdx == trainImgIdx)
            {
                mask.at(i) = 1;
            }
        }

        return std::move(mask);
    }

    void MatchOneToManyDescriptors(size_t queryImgIdx)
    {
        flann_matcher_->clear();
        for (size_t trainImgIdx = 0; trainImgIdx < rgbImgs_.size(); trainImgIdx++)
        {
            flann_matcher_->clear();
            if (trainImgIdx != queryImgIdx) // don't self match
            {
                std::vector<std::vector<cv::DMatch>> knnMatches;
                std::vector<cv::DMatch>              good_matches;
                // find 2 best matches, with distance in increasing order
                flann_matcher_->knnMatch(
                    descriptorsAllImgs_.at(queryImgIdx), descriptorsAllImgs_.at(trainImgIdx), knnMatches, 2);
                for (const auto &matchesPerDescriptor : knnMatches)
                {
                    if (matchesPerDescriptor.at(0).distance < LOWE_MATCH_RATIO * matchesPerDescriptor.at(1).distance)
                    {
                        good_matches.push_back(matchesPerDescriptor.at(0));
                        globalMatchInfo_.at(queryImgIdx)
                            .at(trainImgIdx)
                            .push_back(std::make_pair(matchesPerDescriptor.at(0).queryIdx,
                                                      matchesPerDescriptor.at(0).trainIdx));
                    }
                }

                // Draw
                // cv::Mat drawImg;
                // cv::drawMatches(rgbImgs_.at(queryImgIdx),
                //                 keypointsAllImgs_.at(queryImgIdx),
                //                 rgbImgs_.at(trainImgIdx),
                //                 keypointsAllImgs_.at(trainImgIdx),
                //                 good_matches,
                //                 drawImg,
                //                 cv::Scalar::all(-1),
                //                 cv::Scalar::all(-1),
                //                 std::vector<char>(),
                //                 cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                // cv::imshow("matches", drawImg);
                // cv::moveWindow("matches", 30, 500);
                // cv::waitKey(1000);
            }
            std::cout << "Finished matching image " << queryImgIdx << " to image" << trainImgIdx << std::endl;
        }

        // matplot::cla();
        // matplot::plot(distances, queryIdxs, "o");
        // matplot::hold(matplot::on);
        // matplot::plot(std::vector<float>{bestMatch.distance}, std::vector<int>{bestMatch.imgIdx}, "x");
        // matplot::legend();
        // matplot::grid(matplot::on);
        // matplot::show();
    }

    bool doesMirrorMatchExist(const std::vector<std::pair<int, int>> &trainToQueryMatchedDescriptorPairs,
                              const int                              &queryToTrainMatchedDescriptorPairQueryIdx,
                              const int                              &queryToTrainMatchedDescriptorPairTrainIdx)
    {
        for (const auto &trainToQueryMatchedDescriptorPair : trainToQueryMatchedDescriptorPairs)
        {
            if ((trainToQueryMatchedDescriptorPair.first == queryToTrainMatchedDescriptorPairTrainIdx) &&
                (trainToQueryMatchedDescriptorPair.second == queryToTrainMatchedDescriptorPairQueryIdx))
            {
                return true;
            }
        }
        return false;
    }

    void PruneMatchesByInverseCheck(size_t queryImgIdx)
    {
        size_t trainImgIdx = 0;
        for (std::vector<std::pair<int, int>> &queryToTrainMatchedDescriptorPairs : globalMatchInfo_.at(queryImgIdx))
        {
            std::vector<std::pair<int, int>> prunedDescriptorMatches;
            std::vector<cv::DMatch>          badMatchesForVis;
            for (size_t descIdx = 0; descIdx < queryToTrainMatchedDescriptorPairs.size(); descIdx++)
            {
                auto queryToTrainMatchedDescriptorPair = queryToTrainMatchedDescriptorPairs.at(descIdx);
                // Check if the reverse mapping exists
                auto trainToQueryMatchedDescriptorPairs = globalMatchInfo_.at(trainImgIdx).at(queryImgIdx);
                // create match vector for bad matches visualization

                if (doesMirrorMatchExist(trainToQueryMatchedDescriptorPairs,
                                         queryToTrainMatchedDescriptorPair.first,
                                         queryToTrainMatchedDescriptorPair.second))
                {
                    prunedDescriptorMatches.push_back(queryToTrainMatchedDescriptorPair);
                }
                else
                {
                    badMatchesForVis.push_back(cv::DMatch(
                        queryToTrainMatchedDescriptorPair.first, queryToTrainMatchedDescriptorPair.second, 0));
                }
            }
            queryToTrainMatchedDescriptorPairs = prunedDescriptorMatches; // update with the pruned state

            // ---------- VISUALIZATION ---------- //
            // Show pruned matches
            // cv::Mat drawImg;
            // cv::drawMatches(rgbImgs_.at(queryImgIdx),
            //                 keypointsAllImgs_.at(queryImgIdx),
            //                 rgbImgs_.at(trainImgIdx),
            //                 keypointsAllImgs_.at(trainImgIdx),
            //                 badMatchesForVis,
            //                 drawImg,
            //                 cv::Scalar::all(-1),
            //                 cv::Scalar::all(-1),
            //                 std::vector<char>(),
            //                 cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
            // cv::imshow("matches", drawImg);
            // cv::moveWindow("matches", 30, 500);
            // cv::waitKey(1000);
            // ---------- End of VISUALIZATION ---------- //

            std::cerr << "Pruned " << badMatchesForVis.size() << " matches in mirror checking between imgs "
                      << queryImgIdx << " -> " << trainImgIdx << std::endl;

            trainImgIdx++;
        }
    }

    void PrintMatchedDescriptors()
    {
        for (size_t queryImgIdx = 0; queryImgIdx < rgbImgs_.size(); queryImgIdx++)
        {
            for (size_t trainImgIdx = 0; trainImgIdx < rgbImgs_.size(); trainImgIdx++)
            {
                std::cout << queryImgIdx << "->" << trainImgIdx << " : "
                          << globalMatchInfo_.at(queryImgIdx).at(trainImgIdx).size() << std::endl;
            }
        }
    }

    void ShowLandmarks()
    {
        for (auto landmark : uniqueLandmarks_)
        {
            auto windowLocItr = windowLocations.begin();
            int  windowPosX   = windowLocItr->first;
            int  windowPosY   = windowLocItr->second;
            // show the landmark in each of the viewpoints image
            for (auto descriptor : landmark)
            {
                cv::Mat imgToDraw;
                rgbImgs_.at(descriptor.imgIdx).copyTo(imgToDraw);
                std::cout << descriptor.imgIdx << "->" << descriptor.descIdx << std::endl;
                std::cout << keypointsAllImgs_.at(descriptor.imgIdx).at(descriptor.descIdx).pt << std::endl;
                cv::circle(imgToDraw,
                           keypointsAllImgs_.at(descriptor.imgIdx).at(descriptor.descIdx).pt,
                           _cX / 100,
                           cv::Scalar(0, 0, 250),
                           4);
                cv::imshow(std::to_string(descriptor.imgIdx), imgToDraw);
                cv::moveWindow(std::to_string(descriptor.imgIdx), windowPosX, windowPosY);

                windowLocItr++;
                if (windowLocItr == windowLocations.end())
                    windowLocItr = windowLocations.begin(); // wrap
                windowPosX = windowLocItr->first;
                windowPosY = windowLocItr->second;
            }
            std::cout << "--------------------" << std::endl;

            char c = (char)cv::waitKey();
            if (c == 32) // SPACE
            {
                cv::destroyAllWindows();
                std::this_thread::sleep_for(100ms);
                continue;
            }
        }
    }

    void ShowMatches()
    {
        for (size_t queryImgIdx = 0; queryImgIdx < rgbImgs_.size(); queryImgIdx++)
        {
            for (size_t trainImgIdx = 0; trainImgIdx < rgbImgs_.size(); trainImgIdx++)
            {
                // create match vector for visualization
                std::vector<cv::DMatch> matches;
                for (auto match : globalMatchInfo_.at(queryImgIdx).at(trainImgIdx))
                {
                    // dummy distance since not used for vis
                    matches.push_back(cv::DMatch(match.first, match.second, 0));
                }

                cv::Mat drawImg;
                cv::drawMatches(rgbImgs_.at(queryImgIdx),
                                keypointsAllImgs_.at(queryImgIdx),
                                rgbImgs_.at(trainImgIdx),
                                keypointsAllImgs_.at(trainImgIdx),
                                matches,
                                drawImg,
                                cv::Scalar::all(-1),
                                cv::Scalar::all(-1),
                                std::vector<char>(),
                                cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                cv::imshow("matches", drawImg);
                cv::moveWindow("matches", 30, 500);
                cv::waitKey(1000);
            }
        }
    }

    void FindGlobalMatches()
    {
        FindAllKeypointsAndDescriptors();

        for (size_t queryImgIdx = 0; queryImgIdx < rgbImgs_.size(); queryImgIdx++)
        {
            MatchOneToManyDescriptors(queryImgIdx);
        }

        PrintMatchedDescriptors();
        // ShowMatches();

        // Prune the i->j matches that do not have corresponding j->i
        for (size_t queryImgIdx = 0; queryImgIdx < rgbImgs_.size(); queryImgIdx++)
        {
            PruneMatchesByInverseCheck(queryImgIdx);
        }

        std::cout << "----------------" << std::endl;
        PrintMatchedDescriptors();
        // ShowMatches();

        // Initialize the landmarks vector with a landmark
        std::vector<DescriptorIndentifier> newLandmark{
            DescriptorIndentifier(0, globalMatchInfo_.at(0).at(1).begin()->first),
            DescriptorIndentifier(1, globalMatchInfo_.at(0).at(1).begin()->second)};
        uniqueLandmarks_.push_back(newLandmark);

        FindUniqueLandmarks();

        RemoveLandmarksWithInsufficientViewPoints();

        // ShowLandmarks();
    }

    void RemoveLandmarksWithInsufficientViewPoints()
    {
        decltype(uniqueLandmarks_) prunedLanmarks;

        std::cout << "Size of landmarks before viewpoint pruning: " << uniqueLandmarks_.size() << std::endl;

        for (auto landmark : uniqueLandmarks_)
        {
            if (landmark.size() >= NUM_REQUIRED_VIEWPOINTS_FOR_LANDMARK)
            {
                prunedLanmarks.push_back(landmark);
            }
        }
        uniqueLandmarks_ = prunedLanmarks;

        std::cout << "Size of landmarks after viewpoint pruning: " << uniqueLandmarks_.size() << std::endl;
    }

    // Reduce the pair-wise matched descriptors to unique landmarks across all images
    void FindUniqueLandmarks()
    {
        for (size_t queryImgIdx = 0; queryImgIdx < rgbImgs_.size() - 1; queryImgIdx++)
        {
            for (size_t trainImgIdx = queryImgIdx + 1; trainImgIdx < rgbImgs_.size(); trainImgIdx++)
            {
                // For each matched descriptor pair between queryImg and trainImg
                for (auto queryToTrainMatchedDescriptorPair : globalMatchInfo_.at(queryImgIdx).at(trainImgIdx))
                {
                    // Check if any of the existing landmarks contain query descriptor OR train descriptor
                    bool landmarkFound = false;
                    for (auto &uniqueLandmark : uniqueLandmarks_)
                    {
                        bool queryDescriptorFound = DoesLandmarkHaveThisDescriptor(
                            uniqueLandmark, queryImgIdx, queryToTrainMatchedDescriptorPair.first);
                        bool trainDescriptorFound = DoesLandmarkHaveThisDescriptor(
                            uniqueLandmark, trainImgIdx, queryToTrainMatchedDescriptorPair.second);

                        if (queryDescriptorFound && !trainDescriptorFound)
                        {
                            uniqueLandmark.push_back(
                                DescriptorIndentifier(trainImgIdx, queryToTrainMatchedDescriptorPair.second));
                            landmarkFound = true;
                            break;
                        }
                        else if (!queryDescriptorFound && trainDescriptorFound)
                        {
                            uniqueLandmark.push_back(
                                DescriptorIndentifier(queryImgIdx, queryToTrainMatchedDescriptorPair.first));
                            landmarkFound = true;
                            break;
                        }
                        else if (queryDescriptorFound && trainDescriptorFound)
                        {
                            landmarkFound = true;
                            break;
                        }
                    }
                    if (!landmarkFound)
                    {
                        // Both query and train descriptor not found among existing landmarks, add a new landmark
                        std::vector<DescriptorIndentifier> newLandmark{
                            DescriptorIndentifier(queryImgIdx, queryToTrainMatchedDescriptorPair.first),
                            DescriptorIndentifier(trainImgIdx, queryToTrainMatchedDescriptorPair.second)};
                        uniqueLandmarks_.push_back(newLandmark);

                        // std::cout << "added " << queryImgIdx << ":" << queryToTrainMatchedDescriptorPair.first << " -> "
                        //           << trainImgIdx << ":" << queryToTrainMatchedDescriptorPair.second << std::endl;
                    }
                }
            }
        }
    }

    bool DoesLandmarkHaveThisDescriptor(const std::vector<DescriptorIndentifier> &landmark,
                                        const int                                &imgIdx,
                                        const int                                &descIdx)
    {
        for (const DescriptorIndentifier &descriptorOfLandmark : landmark)
        {
            if ((descriptorOfLandmark.imgIdx == imgIdx) && (descriptorOfLandmark.descIdx == descIdx))
            {
                return true;
            }
        }
        return false;
    }

    void CreateBaDataset()
    {
        // Create a dataset for the Bundle Adjustment

        // camera_idx, unique_3d_pt_idx, current_cam_pix_x, current_cam_pix_y
        // ...
        // repeated num_observation times

        // unique_3d_world_points (in world coordinates)
        // [x,y,z coordinates]
        // ...
        // repeated for number of unique 3d points

        std::vector<std::optional<Eigen::Vector3d>> landmark3dPositions;
        int                                         landmarkIdx     = 0;
        int                                         numObservations = 0;

        // Write the camera observations of the landmarks
        for (auto landmark : uniqueLandmarks_)
        {
            // Use the 1st descriptor's viewpoint that sees this landmark while doing 3d projection
            auto imgIdx   = landmark.begin()->imgIdx;
            auto keypoint = keypointsAllImgs_.at(imgIdx).at(landmark.begin()->descIdx);
            // get the pose of this camera from the odometry
            auto            cameraPose = cameraPoses_.at(imgIdx);
            Eigen::Matrix3d cameraRotation =
                Eigen::Quaterniond(cameraPose.qw, cameraPose.qx, cameraPose.qy, cameraPose.qz).toRotationMatrix();
            Eigen::Vector3d                cameraTranslation(cameraPose.x, cameraPose.y, cameraPose.z);
            std::optional<Eigen::Vector3d> landmarkInWorldCoord =
                projectKeypointTo3d(imgIdx, keypoint, cameraRotation, cameraTranslation);
            landmark3dPositions.push_back(landmarkInWorldCoord);

            if (landmark3dPositions.back())
            {
                for (auto descriptor : landmark)
                {
                    imgIdx   = descriptor.imgIdx;
                    keypoint = keypointsAllImgs_.at(imgIdx).at(descriptor.descIdx);
                    numObservations++;
                    baDataOutStream_ << imgIdx << " " << landmarkIdx << " " << keypoint.pt.x << " " << keypoint.pt.y
                                     << std::endl;
                }
                landmarkIdx++;
            }
        }

        // Write the camera poses
        for (const RoboticsPose &cameraPose : cameraPoses_)
        {
            baDataOutStream_ << cameraPose.x << " " << cameraPose.y << " " << cameraPose.z << " " << cameraPose.qx
                             << " " << cameraPose.qy << " " << cameraPose.qz << " " << cameraPose.qw << std::endl;
        }

        // Write the 3D poses of the landmarks
        for (auto landmarkPosition : landmark3dPositions)
        {
            if (landmarkPosition)
            {
                baDataOutStream_ << (*landmarkPosition)(0) << " " << (*landmarkPosition)(1) << " "
                                 << (*landmarkPosition)(2) << std::endl;
            }
        }

        // Add header
        baDataOutStream_.close();
        std::ifstream     tempIn(BA_DATA_PATH);
        std::stringstream payloadBuffer;
        payloadBuffer << tempIn.rdbuf();
        baDataOutStream_.open(BA_DATA_PATH);
        baDataOutStream_ << "num_cameras num_3d_landmarks num_observations" << std::endl;
        baDataOutStream_ << rgbImgs_.size() << " " << landmarkIdx << " " << numObservations
                         << std::endl; // to be replaced
        baDataOutStream_ << payloadBuffer.rdbuf();
    }

    ~FeatureMatcher()
    {
        std::cout << "Destructed\n";
    }

    std::optional<Eigen::Vector3d> projectKeypointTo3d(const int             &imgIdx,
                                                       const cv::KeyPoint    &keypoint,
                                                       const Eigen::Matrix3d &rot   = Eigen::Matrix3d::Identity(),
                                                       const Eigen::Vector3d &trans = Eigen::Vector3d::Zero())
    {
        std::vector<Eigen::Vector3d> o3d_points;
        std::vector<Eigen::Vector3d> o3d_colors;

        float  kpX   = keypoint.pt.x;
        float  kpY   = keypoint.pt.y;
        double depth = depthImgs_.at(imgIdx).at<double>(kpY, kpX);
        // double depth = depthImage.at<double>(ii, jj);
        if ((depth > 0) && (depth < DEPTH_THRESHOLD)) // its possible that stereo depth returns 0
        {
            double          x_world = (kpX - _cX) * depth / _fX;
            double          y_world = (kpY - _cY) * depth / _fY;
            double          z_world = depth;
            Eigen::Vector3d pt_in_cam_frame(-x_world, -y_world, z_world);
            Eigen::Vector3d pt_in_world_frame = pt_in_cam_frame;
            transformPoint(rot, trans, pt_in_world_frame);

            return pt_in_world_frame;
        }
        else
        {
            return {};
        }
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

  private:
    // camera intrinsics
    double  _cX, _cY, _fX, _fY;
    cv::Mat K_;

    open3d::visualization::Visualizer             vis;
    std::shared_ptr<open3d::geometry::PointCloud> o3d_cloud;

    // current camera transformation matrix (Translation + Rotation)
    Eigen::Matrix4f                T_;
    cv::Ptr<cv::Feature2D>         orb_;
    cv::Ptr<cv::Feature2D>         sift_;
    cv::Ptr<cv::DescriptorMatcher> flann_matcher_;

    std::vector<cv::Mat> rgbImgs_;
    std::vector<cv::Mat> depthImgs_;

    std::vector<RoboticsPose> cameraPoses_;

    std::vector<std::vector<cv::KeyPoint>> keypointsAllImgs_;
    std::vector<cv::Mat>                   descriptorsAllImgs_;

    // (num_imgs - 1) x(num_imgs - 1) x(num_matches_for_i_j_image_pair)x(source_descriptor_index,target_descriptor_index)
    // globalMatchInfo_.at(queryImgIdx).at(targetImgIdx) --> gives a vector of matched descriptors between query and target image, as (queryImgDescriptorIdx,targetImgDescriptorIdx) pair
    std::vector<std::vector<std::vector<std::pair<int, int>>>> globalMatchInfo_;

    std::vector<std::vector<DescriptorIndentifier>> uniqueLandmarks_; // a landmark is a set set of Descriptors

    std::ofstream baDataOutStream_;
};
} // namespace sfm