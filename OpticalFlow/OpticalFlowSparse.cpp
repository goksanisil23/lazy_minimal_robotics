#include "OpticalFlowSparse.h"

OpticalFlowSparse::OpticalFlowSparse()
{
    featureDetector_ = cv::GFTTDetector::create(
        MAX_DETECTOR_CORNERS, DETECTOR_QUALITY_LEVEL, DETECTOR_MIN_PIX_DIST_KPS, DETECTOR_BLOCK_SIZE);

    // Image pyramic config
    pyrScales_.at(0) = 1.0;
    for (size_t i = 1; i < NUM_PYRAMID_LEVELS; i++)
    {
        pyrScales_.at(i) = pyrScales_.at(i - 1) * PYRAMID_SCALE_FACTOR;
    }
}

OpticalFlowSparse::OpticalFlowSparse(const OpticalFlowConfig &config) : OpticalFlowSparse()
{
    config_ = std::move(config);
}

void OpticalFlowSparse::Detect(const cv::Mat &img1, std::vector<cv::KeyPoint> &kpsImg1Out)
{
    auto t0 = time_util::chronoNow();
    featureDetector_->detect(img1, kpsImg1Out);
    auto t1 = time_util::chronoNow();
    time_util::showTimeDuration(t1, t0, "Feature detection: ");
}

void OpticalFlowSparse::Track(const cv::Mat                   &img1,
                              const cv::Mat                   &img2,
                              const std::vector<cv::KeyPoint> &kpsImg1In,
                              std::vector<cv::KeyPoint>       &kpsImg2Out,
                              std::vector<bool>               &isFlowOkOut)
{
    if (img1.channels() > 1)
        cv::cvtColor(img1, image1_, cv::COLOR_BGR2GRAY);
    else
        image1_ = img1;
    if (img2.channels() > 1)
        cv::cvtColor(img2, image2_, cv::COLOR_BGR2GRAY);
    else
        image2_ = img2;

    switch (config_.level)
    {

    case OpticalFlowConfig::LEVEL::SINGLE:
    {
        kpsImg2Out.resize(kpsImg1In.size());
        isFlowOkOut.resize(kpsImg1In.size());

        auto t0 = time_util::chronoNow();
        ComputeFlowSparse(kpsImg1In, kpsImg2Out, isFlowOkOut, OpticalFlowConfig::USE_INIT_GUESS::FALSE);
        auto t1 = time_util::chronoNow();
        time_util::showTimeDuration(t1, t0, "Single layer oflow: ");

        ShowSparseFlow(img2, kpsImg1In, kpsImg2Out, isFlowOkOut);
        break;
    }

    case OpticalFlowConfig::LEVEL::MULTI:
    {

        kpsImg2Out.resize(kpsImg1In.size());
        isFlowOkOut.resize(kpsImg1In.size());

        auto t0 = time_util::chronoNow();
        // 1) Create the image pyramids
        std::vector<cv::Mat> img1Pyra(NUM_PYRAMID_LEVELS);
        std::vector<cv::Mat> img2Pyra(NUM_PYRAMID_LEVELS);
        img1Pyra.at(0) = image1_;
        img2Pyra.at(0) = image2_;
        for (size_t pyLvlIdx = 1; pyLvlIdx < NUM_PYRAMID_LEVELS; pyLvlIdx++)
        {
            cv::Mat img1Scaled, img2Scaled;
            cv::resize(img1Pyra.at(pyLvlIdx - 1),
                       img1Scaled,
                       cv::Size(img1Pyra.at(pyLvlIdx - 1).cols * PYRAMID_SCALE_FACTOR,
                                img1Pyra.at(pyLvlIdx - 1).rows * PYRAMID_SCALE_FACTOR));
            cv::resize(img2Pyra.at(pyLvlIdx - 1),
                       img2Scaled,
                       cv::Size(img2Pyra.at(pyLvlIdx - 1).cols * PYRAMID_SCALE_FACTOR,
                                img2Pyra.at(pyLvlIdx - 1).rows * PYRAMID_SCALE_FACTOR));
            img1Pyra.at(pyLvlIdx) = img1Scaled;
            img2Pyra.at(pyLvlIdx) = img2Scaled;
        }
        auto t1 = time_util::chronoNow();
        // ShowImagePyramid(img1Pyra);

        // 2) Recalculate the keypoints in the coarsest (top) level
        // since we go from coarse to fine in optical flow
        std::vector<cv::KeyPoint> kpsImg1Pyr(kpsImg1In); // keypoints of image 1 at the current pyramid level
        for (auto &kpImg1 : kpsImg1Pyr)
        {
            kpImg1.pt *= pyrScales_.back();
        }
        std::vector<cv::KeyPoint> kpsImg2Pyr(kpsImg1Pyr);

        // 3) Run Optical Flow from top to bottom layer
        for (int pyrLvlIdx = NUM_PYRAMID_LEVELS - 1; pyrLvlIdx >= 0; pyrLvlIdx--)
        {
            // Update the source and destination images for optical flow
            image1_ = img1Pyra.at(pyrLvlIdx);
            image2_ = img2Pyra.at(pyrLvlIdx);
            // Run the oflow
            ComputeFlowSparse(kpsImg1Pyr, kpsImg2Pyr, isFlowOkOut, OpticalFlowConfig::USE_INIT_GUESS::TRUE);

            // Upscale the keypoints since we're going down pyramid layer
            if (pyrLvlIdx != 0)
            {
                for (size_t kpIdx = 0; kpIdx < kpsImg1Pyr.size(); kpIdx++)
                {
                    kpsImg1Pyr.at(kpIdx).pt /= PYRAMID_SCALE_FACTOR;
                    kpsImg2Pyr.at(kpIdx).pt /= PYRAMID_SCALE_FACTOR;
                }
            }
        }

        kpsImg2Out = kpsImg2Pyr;

        auto t2 = time_util::chronoNow();

        time_util::showTimeDuration(t1, t0, "Pyramid creation : ");
        time_util::showTimeDuration(t2, t1, "Pyramid oflow    : ");

        // ShowSparseFlow(img2, kpsImg1In, kpsImg2Out, isFlowOkOut);

        break;
    }
    }
}

// Given frames at [k] and [k+1] and keypoints at [k], find where those keypoints would shift in image k+1.
// This function uses Gauss-Newton method to minimize the pixel intensity residual in a small image patch
// Since optical flow is built on the assumption that the pixel intensities of a unique point remains the same from k to k+1,
// this function iteratively tries to find Δx & Δy for the keypoint at frame [k], that would result with same keypoint at k+1.
void OpticalFlowSparse::ComputeFlowSparse(const std::vector<cv::KeyPoint>  &kpsImg1In,
                                          std::vector<cv::KeyPoint>        &kpsImg2Out,
                                          std::vector<bool>                &isFlowOkOut,
                                          OpticalFlowConfig::USE_INIT_GUESS useInitGuess)
{
    omp_set_num_threads(4);
#pragma omp parallel for
    for (size_t kpIdx = 0; kpIdx < kpsImg1In.size(); kpIdx++)
    {
        // const auto &kp{kpsImg1.at(kpIdx)};
        const cv::KeyPoint kp = kpsImg1In.at(kpIdx);
        float              dx{0}, dy{0}; // deltas represent the flow the will be estimated
        if (useInitGuess == OpticalFlowConfig::USE_INIT_GUESS::TRUE)
        {
            dx = kpsImg2Out.at(kpIdx).pt.x - kpsImg1In.at(kpIdx).pt.x;
            dy = kpsImg2Out.at(kpIdx).pt.y - kpsImg1In.at(kpIdx).pt.y;
        }

        float cost{0}, lastCost{0};
        bool  success{true}; // tracking of the point is successfull

        // Gauss-Newton iterations
        Eigen::Matrix2d H    = Eigen::Matrix2d::Zero(); // Hessian
        Eigen::Vector2d bias = Eigen::Vector2d::Zero(); // cumulative bias for the image patch
        Eigen::Vector2d J;                              // Jacobian

        for (int iter = 0; iter < NUM_GAUSS_NEWTON_ITERS; iter++)
        {
            if (!config_.useInvFormulation)
            {
                H    = Eigen::Matrix2d::Zero();
                bias = Eigen::Vector2d::Zero();
            }
            else
            {
                bias = Eigen::Vector2d::Zero();
            }

            cost = 0; // cumulative cost for the image patch
            // Compute Jacobian and cost for the image patch
            for (int x = -HALF_NEIGHBORHOOD_SIZE; x < HALF_NEIGHBORHOOD_SIZE; x++)
            {
                for (int y = -HALF_NEIGHBORHOOD_SIZE; y < HALF_NEIGHBORHOOD_SIZE; y++)
                {
                    // Residual for Jacobian
                    // Observation: Intensity of the pixel in current image
                    // Estimation model: Intensity of the of the displaced pixel (with current delta estimation) in the next image
                    float residual = GetPixelValue(image1_, kp.pt.x + x, kp.pt.y + y) -
                                     GetPixelValue(image2_, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    if (!config_.useInvFormulation)
                    {
                        J = -1.0 *
                            Eigen::Vector2d(0.5 * (GetPixelValue(image2_, (kp.pt.x + x) + dx + 1, (kp.pt.y + y) + dy) -
                                                   GetPixelValue(image2_, (kp.pt.x + x) + dx - 1, (kp.pt.y + y) + dy)),
                                            0.5 * (GetPixelValue(image2_, (kp.pt.x + x) + dx, (kp.pt.y + y) + dy + 1) -
                                                   GetPixelValue(image2_, (kp.pt.x + x) + dx, (kp.pt.y + y) + dy - 1)));
                    }
                    else if (iter == 0)
                    {
                        J = -1.0 * Eigen::Vector2d(0.5 * (GetPixelValue(image1_, kp.pt.x + x + 1, kp.pt.y + y) -
                                                          GetPixelValue(image1_, kp.pt.x + x - 1, kp.pt.y + y)),
                                                   0.5 * (GetPixelValue(image1_, kp.pt.x + x, kp.pt.y + y + 1) -
                                                          GetPixelValue(image1_, kp.pt.x + x, kp.pt.y + y - 1)));
                    }

                    bias += -residual * J;
                    cost += residual * residual;
                    if (!config_.useInvFormulation || iter == 0)
                        H += J * J.transpose();
                }
            }

            // Solve for the entire image patch
            // Eigen::Vector2d update = H.ldlt().solve(bias);
            Eigen::Vector2d update = H.colPivHouseholderQr().solve(bias);
            if (std::isnan(update[0]) || std::isnan(update[1]))
            {
                std::cerr << "NAN update, cancelling flow for this keypoint" << std::endl;
                success = false;
                break;
            }
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

// // Get pixel value with Bilinear Interpolation
inline float OpticalFlowSparse::GetPixelValue(const cv::Mat &img, float x, float y)
{
    // boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= (img.cols - 1))
        x = (img.cols - 2);
    if (y >= (img.rows - 1))
        y = (img.rows - 2);

    float xx   = x - std::floor(x);
    float yy   = y - std::floor(y);
    int   x_a1 = std::min(img.cols - 1, static_cast<int>(x) + 1);
    int   y_a1 = std::min(img.rows - 1, static_cast<int>(y) + 1);

    return (1.0f - xx) * (1.0f - yy) * static_cast<float>(img.at<uchar>(y, x)) +
           xx * (1.0f - yy) * static_cast<float>(img.at<uchar>(y, x_a1)) +
           (1.0f - xx) * yy * static_cast<float>(img.at<uchar>(y_a1, x)) +
           xx * yy * static_cast<float>(img.at<uchar>(y_a1, x_a1));
}

void OpticalFlowSparse::ShowSparseFlow(const cv::Mat                   &img2,
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
            cv::line(img2Bgr, kpsImg2[kpIdx].pt, kpsImg1[kpIdx].pt, cv::Scalar(0, 250, 0), 3);
        }
    }
    cv::imshow("oflow", img2Bgr);
    cv::waitKey(0);
}

void OpticalFlowSparse::ShowImagePyramid(const std::vector<cv::Mat> &imagePyr)
{
    // Upscale all the images in the pyramid to same size
    // for visualization
    const cv::Size       targetSize(imagePyr.at(0).cols, imagePyr.at(0).rows);
    std::vector<cv::Mat> pyrUpscaled(imagePyr.size());
    pyrUpscaled.at(0) = imagePyr.at(0);
    cv::imshow("level_" + std::to_string(0), pyrUpscaled.at(0));
    for (size_t pyrLvlIdx = 1; pyrLvlIdx < imagePyr.size(); pyrLvlIdx++)
    {
        cv::resize(imagePyr.at(pyrLvlIdx), pyrUpscaled.at(pyrLvlIdx), targetSize);
        cv::imshow("level_" + std::to_string(pyrLvlIdx), pyrUpscaled.at(pyrLvlIdx));
    }
}