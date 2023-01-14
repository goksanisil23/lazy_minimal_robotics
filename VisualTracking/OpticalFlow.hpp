#pragma once

#include <functional>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <omp.h>
#include <opencv4/opencv2/opencv.hpp>

// #include "TimeUtil.h"

namespace
{
// determines the size of the window around a pixel point to calculate the flow field
constexpr int    HALF_NEIGHBORHOOD_SIZE{4};
constexpr int    MAX_DETECTOR_CORNERS{500};
constexpr double DETECTOR_QUALITY_LEVEL{0.01};
constexpr int    NUM_GAUSS_NEWTON_ITERS{10};
constexpr float  GAUSS_NEWTON_CONV_THRESH_NORM{1e-2};
} // namespace

class OpticalFlow
{
  public:
    struct OpticalFlowConfig
    {
        enum LEVEL
        {
            SINGLE = (1 << 0),
            MULTI  = (1 << 1)
        };

        enum MODE
        {
            SPARSE = (1 << 0),
            DENSE  = (1 << 1)
        };

        bool  useInvFormulation{false};
        LEVEL level{LEVEL::SINGLE};
        MODE  mode{MODE::SPARSE};
        int   halfWinSize{HALF_NEIGHBORHOOD_SIZE};
    };

    OpticalFlow()
    {
        featureDetector_ = cv::GFTTDetector::create(MAX_DETECTOR_CORNERS, DETECTOR_QUALITY_LEVEL, 20);
    }

    OpticalFlow(const OpticalFlowConfig &config) : OpticalFlow()
    {
        config_ = std::move(config);
    }

    void Step(const cv::Mat &img1, const cv::Mat &img2)
    {
        if (img1.channels() > 1)
            cv::cvtColor(img1, image1_, cv::COLOR_BGR2GRAY);
        else
            img1.copyTo(image1_);
        if (img2.channels() > 1)
            cv::cvtColor(img2, image2_, cv::COLOR_BGR2GRAY);
        else
            img2.copyTo(image2_);

        switch (config_.mode)
        {
        case OpticalFlowConfig::MODE::SPARSE:
        {
            // auto t0 = time_util::chronoNow();

            std::vector<cv::KeyPoint> kpsImg1;
            featureDetector_->detect(img1, kpsImg1);

            // auto t1 = time_util::chronoNow();

            std::vector<cv::KeyPoint> kpsImg2(kpsImg1.size());
            std::vector<bool>         isFlowOk(kpsImg1.size());

            // cv::Range range(0, kpsImg1.size());

            // ********** WITH PARALLEL FOR LOOP ********** //
            // auto functorOflow = [&](const cv::Range &range) { ComputeFlowSparse(range, kpsImg1, kpsImg2, isFlowOk); };
            // cv::parallel_for_(cv::Range(0, kpsImg1.size()), functorOflow);
            // auto functorOflow = [&](const cv::Range &range) { ComputeFlowSparse(range); };
            // cv::parallel_for_(cv::Range(0, kpsImg1.size()), functorOflow);
            // cv::parallel_for_(cv::Range(0, kpsImg1.size()),
            //                   std::bind(&OpticalFlow::ComputeFlowSparse, this, std::placeholders::_1));
            // cv::parallel_for_(cv::Range(0, kpsImg1.size()),
            //                   [&](const cv::Range &range)
            //                   {
            //                       for (int i = range.start; i < range.end; i++)
            //                           ComputeFlowSparseSingle(i);
            //                   });

            //             omp_set_num_threads(3);
            // #pragma omp parallel for
            //             for (int i = 0; i < kpsImg1.size(); i++)
            //             {
            //                 ComputeFlowSparseSingle(i);
            //             }

            // ********** WITHOUT PARALLEL FOR LOOP ********** //
            // auto range = cv::Range(0, kpsImg1.size());
            // for (int i = range.start; i < range.end; i++)
            //     ComputeFlowSparseSingle(i);
            ComputeFlowSparse(kpsImg1, kpsImg2, isFlowOk);

            // auto t2 = time_util::chronoNow();

            // time_util::showTimeDuration(t1, t0, "Feature detection: ");
            // time_util::showTimeDuration(t2, t1, "Single layer oflow: ");

            std::for_each(allUpdates.begin(), allUpdates.end(), [](Eigen::Vector2d updt) { std::cout << updt << " "; });
            std::cout << "\n-------------------- keypoints --------------" << std::endl;
            std::for_each(
                kpsImg2.begin(), kpsImg2.end(), [](cv::KeyPoint kp) { std::cout << kp.pt.x << " " << kp.pt.y << " "; });

            ShowSparseFlow(img2, kpsImg1, kpsImg2, isFlowOk);
            break;
        }

        case OpticalFlowConfig::MODE::DENSE:
        {
            break;
        }
        }
    }

    // Given frames at [k] and [k+1] and keypoints at [k], find where those keypoints would shift in image k+1.
    // This function uses Gauss-Newton method to minimize the pixel intensity residual in a small image patch
    // Since optical flow is built on the assumption that the pixel intensities of a unique point remains the same from k to k+1,
    // this function iteratively tries to find Δx & Δy for the keypoint at frame [k], that would result with same keypoint at k+1.
    void ComputeFlowSparse(const std::vector<cv::KeyPoint> &kpsImg1In,
                           std::vector<cv::KeyPoint>       &kpsImg2Out,
                           std::vector<bool>               &isFlowOkOut)
    {
        omp_set_num_threads(3);
#pragma omp parallel for
        for (size_t kpIdx = 0; kpIdx < kpsImg1In.size(); kpIdx++)
        {
            // const auto &kp{kpsImg1.at(kpIdx)};
            const cv::KeyPoint kp = kpsImg1In.at(kpIdx);
            float              dx{0}, dy{0}; // deltas represent the flow the will be estimated

            float cost{0}, lastCost{0};
            bool  success{true}; // tracking of the point is successfull

            // Gauss-Newton iterations
            Eigen::Matrix2d H    = Eigen::Matrix2d::Zero(); // Hessian
            Eigen::Vector2d bias = Eigen::Vector2d::Zero(); // bias
            Eigen::Vector2d J;                              // Jacobian

            for (int iter = 0; iter < NUM_GAUSS_NEWTON_ITERS; iter++)
            {
                H    = Eigen::Matrix2d::Zero();
                bias = Eigen::Vector2d::Zero(); // cumulative bias for the image patch
                cost = 0;                       // cumulative cost for the image patch
                // Compute Jacobian and cost for the image patch
                for (int x = -HALF_NEIGHBORHOOD_SIZE; x < HALF_NEIGHBORHOOD_SIZE; x++)
                {
                    for (int y = -HALF_NEIGHBORHOOD_SIZE; y < HALF_NEIGHBORHOOD_SIZE; y++)
                    {
                        // Residual for Jacobian
                        // Observation: Intensity of the pixel in current image
                        // Estimation model: Intensity of the of the displaced pixel (with current delta estimation) in the next image
                        // std::cout << "outside: " << kp.pt.x + x << "  " << kp.pt.y + y << std::endl;
                        // std::cout << "update: " << dx << " " << dy << std::endl;
                        float residual = GetPixelValue(image1_, kp.pt.x + x, kp.pt.y + y) -
                                         GetPixelValue(image2_, kp.pt.x + x + dx, kp.pt.y + y + dy);
                        // std::cout << "pixel val 2: " << GetPixelValue(image2_, kp.pt.x + x + dx, kp.pt.y + y + dy)
                        //           << std::endl;
                        J = -1.0 *
                            Eigen::Vector2d(0.5 * (GetPixelValue(image2_, (kp.pt.x + x) + dx + 1, (kp.pt.y + y) + dy) -
                                                   GetPixelValue(image2_, (kp.pt.x + x) + dx - 1, (kp.pt.y + y) + dy)),
                                            0.5 * (GetPixelValue(image2_, (kp.pt.x + x) + dx, (kp.pt.y + y) + dy + 1) -
                                                   GetPixelValue(image2_, (kp.pt.x + x) + dx, (kp.pt.y + y) + dy - 1)));

                        bias += -residual * J;
                        cost += residual * residual;
                        H += J * J.transpose();
                    }
                }

                // Solve for the entire image patch
                Eigen::Vector2d update = H.ldlt().solve(bias);
                if (std::isnan(update[0]) || std::isnan(update[1]))
                {
                    std::cerr << "update is nan" << std::endl;
                    success = false;
                    break;
                }
                // allUpdates.push_back(update);
                // Stop as soon as we are not converging anymore
                if (iter > 0 && cost > lastCost)
                {
                    break;
                }

                // Update dx dy
                dx += update[0];
                dy += update[1];
                lastCost = cost;
                success  = true;

                if (update.norm() < GAUSS_NEWTON_CONV_THRESH_NORM)
                {
                    break;
                }
            }

            kpsImg2Out.at(kpIdx).pt = kp.pt + cv::Point2f(dx, dy);
            isFlowOkOut.at(kpIdx)   = success;
        }
    }

    // Get pixel value with Bilinear Interpolation
    // inline float GetPixelValue(const cv::Mat &img, float x, float y)
    // {
    //     // std::cout << "boundary: " << x << " " << y << std::endl;
    //     // boundary check
    //     if (x < 0)
    //         x = 0;
    //     if (y < 0)
    //         y = 0;
    //     if (x >= (img.cols - 1))
    //         x = (img.cols - 2);
    //     if (y >= (img.rows - 1))
    //         y = (img.rows - 2);

    //     // std::cout << "after: " << x << " " << y << std::endl;

    //     float xx   = x - std::floor(x);
    //     float yy   = y - std::floor(y);
    //     int   x_a1 = std::min(img.cols - 1, static_cast<int>(x) + 1);
    //     int   y_a1 = std::min(img.rows - 1, static_cast<int>(y) + 1);

    //     // std::cout << x << " " << xx << " " << x_a1 << " " << y << " " << yy << " " << y_a1 << std::endl;

    //     return (1.0f - xx) * (1.0f - yy) * static_cast<float>(img.at<uchar>(y, x)) +
    //            xx * (1.0f - yy) * static_cast<float>(img.at<uchar>(y, x_a1)) +
    //            (1.0f - xx) * yy * static_cast<float>(img.at<uchar>(y_a1, x)) +
    //            xx * yy * static_cast<float>(img.at<uchar>(y_a1, x_a1));
    // }

    inline float GetPixelValue(const cv::Mat &img, float x, float y)
    {
        int x1 = (int)x;
        int y1 = (int)y;
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        // Check if coordinates are out of frame
        if (x < 0)
            x1 = 0, x2 = 0;
        if (y < 0)
            y1 = 0, y2 = 0;
        if (x > img.cols - 1)
            x1 = img.cols - 1, x2 = img.cols - 1;
        if (y > img.rows - 1)
            y1 = img.rows - 1, y2 = img.rows - 1;

        float a = x - x1;
        float b = y - y1;

        float top    = (1 - a) * img.at<uchar>(y1, x1) + a * img.at<uchar>(y1, x2);
        float bottom = (1 - a) * img.at<uchar>(y2, x1) + a * img.at<uchar>(y2, x2);

        return (1 - b) * top + b * bottom;
    }

    void ShowSparseFlow(const cv::Mat                   &img2,
                        const std::vector<cv::KeyPoint> &kpsImg1,
                        const std::vector<cv::KeyPoint> &kpsImg2,
                        const std::vector<bool>         &isFlowOk)
    {
        cv::Mat img2Bgr;
        if (img2.channels() == 1)
            cv::cvtColor(img2, img2Bgr, cv::COLOR_GRAY2BGR);
        else
            img2.copyTo(img2Bgr);
        for (size_t kpIdx = 0; kpIdx < kpsImg2.size(); kpIdx++)
        {
            if (isFlowOk.at(kpIdx))
            {
                cv::circle(img2Bgr, kpsImg2[kpIdx].pt, 2, cv::Scalar(0, 250, 0), 2);
                cv::line(img2Bgr, kpsImg1[kpIdx].pt, kpsImg2[kpIdx].pt, cv::Scalar(0, 250, 0));
            }
        }
        cv::imshow("oflow", img2Bgr);
        cv::waitKey(0);
    }

    OpticalFlowConfig      config_;
    cv::Mat                image1_;
    cv::Mat                image2_; // previous and current image that the flow is being computed for
    cv::Ptr<cv::Feature2D> featureDetector_;

    // TODO: remove
    // std::vector<cv::KeyPoint> kpsImg1;
    // std::vector<cv::KeyPoint> kpsImg2;
    // std::vector<bool>         isFlowOk;
    std::vector<Eigen::Vector2d> allUpdates;

  private:
};