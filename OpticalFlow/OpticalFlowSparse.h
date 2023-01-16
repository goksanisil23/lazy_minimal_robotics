#pragma once

#include <functional>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <omp.h>
#include <opencv4/opencv2/opencv.hpp>

#include "TimeUtil.h"

class OpticalFlowSparse
{
  public:
    // ------------ Class Constants  ------------ //

    // determines the size of the window around a pixel point to calculate the flow field
    static constexpr int    DETECTOR_BLOCK_SIZE{8};
    static constexpr int    HALF_NEIGHBORHOOD_SIZE{DETECTOR_BLOCK_SIZE};
    static constexpr int    MAX_DETECTOR_CORNERS{100};
    static constexpr double DETECTOR_QUALITY_LEVEL{0.3};
    static constexpr double DETECTOR_MIN_PIX_DIST_KPS{20.0};
    static constexpr int    NUM_GAUSS_NEWTON_ITERS{10};
    static constexpr float  GAUSS_NEWTON_CONV_THRESH_NORM{1e-2};
    static constexpr int    NUM_PYRAMID_LEVELS{4};
    static constexpr float  PYRAMID_SCALE_FACTOR{0.5}; // how much scaling down from one layer to another

    // ------------ Data structure definitions ------------ //
    struct OpticalFlowConfig
    {
        enum LEVEL
        {
            SINGLE = (1 << 0),
            MULTI  = (1 << 1)
        };

        enum USE_INIT_GUESS
        {
            TRUE  = (1 << 0),
            FALSE = (1 << 1)
        };

        bool  useInvFormulation{true};
        LEVEL level{LEVEL::MULTI};
        int   halfWinSize{HALF_NEIGHBORHOOD_SIZE};
    };

    // ------------ Member Functions ------------ //

    OpticalFlowSparse();
    OpticalFlowSparse(const OpticalFlowConfig &config);

    void         Detect(const cv::Mat &img1, std::vector<cv::KeyPoint> &kpsImg1Out);
    void         Track(const cv::Mat                   &img1,
                       const cv::Mat                   &img2,
                       const std::vector<cv::KeyPoint> &kpsImg1In,
                       std::vector<cv::KeyPoint>       &kpsImg2Out,
                       std::vector<bool>               &isFlowOkOut);
    void         ComputeFlowSparse(const std::vector<cv::KeyPoint>  &kpsImg1In,
                                   std::vector<cv::KeyPoint>        &kpsImg2Out,
                                   std::vector<bool>                &isFlowOkOut,
                                   OpticalFlowConfig::USE_INIT_GUESS useInitGuess);
    inline float GetPixelValue(const cv::Mat &img, float x, float y);
    void         ShowSparseFlow(const cv::Mat                   &img2,
                                const std::vector<cv::KeyPoint> &kpsImg1,
                                const std::vector<cv::KeyPoint> &kpsImg2,
                                const std::vector<bool>         &isFlowOk);
    void         ShowImagePyramid(const std::vector<cv::Mat> &imagePyr);

  private:
    // ------------ Member Variables ------------ //
    OpticalFlowConfig      config_;
    cv::Mat                image1_;
    cv::Mat                image2_; // previous and current image that the flow is being computed for
    cv::Ptr<cv::Feature2D> featureDetector_;

    std::array<float, NUM_PYRAMID_LEVELS> pyrScales_;
};