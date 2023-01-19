#pragma once

#include "Tracker.h"

#include "TimeUtil.h"

class KalmanTracker : public VisualTracker
{
  public:
    struct Config
    {
        int stateSize{6}; // c_x,c_y,v_x,v_y,w,h (center of bbox, velocity of bbox, height, width)
        int measSize{4};  // c_x, c_y, w, h
        int ctrlSize{0};  // we dont have any control parameter in this case
    };

    void Init(const cv::Mat &refImg, const cv::Rect &refBbox) override
    {
        kalmanFilter_ =
            std::make_unique<cv::KalmanFilter>(config_.stateSize, config_.measSize, config_.ctrlSize, CV_32F);

        // We will add dt per iteration
        // State transition matrix
        // [ 1 0 dt 0  0 0 ]
        // [ 0 1 0  dt 0 0 ]
        // [ 0 0 1  0  0 0 ]
        // [ 0 0 0  1  0 0 ]
        // [ 0 0 0  0  1 0 ]
        // [ 0 0 0  0  0 1 ]
        cv::setIdentity(kalmanFilter_->transitionMatrix);

        // Measurement Matrix
        // [ 1 0 0 0 0 0 ]
        // [ 0 1 0 0 0 0 ]
        // [ 0 0 0 0 1 0 ]
        // [ 0 0 0 0 0 1 ]
        kalmanFilter_->measurementMatrix               = cv::Mat::zeros(measSize, stateSize, CV_32F);
        kalmanFilter_->measurementMatrix.at<float>(0)  = 1.0f;
        kalmanFilter_->measurementMatrix.at<float>(7)  = 1.0f;
        kalmanFilter_->measurementMatrix.at<float>(16) = 1.0f;
        kalmanFilter_->measurementMatrix.at<float>(23) = 1.0f;

        // Process Noise Covariance Matrix Q
        // [ E_c_x   0   0     0     0    0  ]
        // [ 0    E_c_y  0     0     0    0  ]
        // [ 0    0     E_v_x  0     0    0  ]
        // [ 0    0     0     E_v_y  0    0  ]
        // [ 0    0     0     0     E_w   0  ]
        // [ 0    0     0     0     0    E_h ]
        kalmanFilter_->processNoiseCov.at<float>(0)  = 1e-2;
        kalmanFilter_->processNoiseCov.at<float>(7)  = 1e-2;
        kalmanFilter_->processNoiseCov.at<float>(14) = 5.0f;
        kalmanFilter_->processNoiseCov.at<float>(21) = 5.0f;
        kalmanFilter_->processNoiseCov.at<float>(28) = 1e-2;
        kalmanFilter_->processNoiseCov.at<float>(35) = 1e-2;

        // Process
        kalmanFilter_->errorCovPre.at<float>(0)  = 1;
        kalmanFilter_->errorCovPre.at<float>(7)  = 1;
        kalmanFilter_->errorCovPre.at<float>(14) = 1;
        kalmanFilter_->errorCovPre.at<float>(21) = 1;
        kalmanFilter_->errorCovPre.at<float>(28) = 1;
        kalmanFilter_->errorCovPre.at<float>(35) = 1;

        // Measures Noise Covariance Matrix R
        cv::setIdentity(kalmanFilter_->measurementNoiseCov, cv::Scalar(1e-1));

        // Initialize the filter
        cv::Point2f bboxCenter                = cv::Point2f(refBbox.tl() + refBbox.br()) / 2.0;
        kalmanFilter_->statePost.at<float>(0) = bboxCenter.x;
        kalmanFilter_->statePost.at<float>(1) = bboxCenter.y;
        kalmanFilter_->statePost.at<float>(2) = 0;
        kalmanFilter_->statePost.at<float>(3) = 0;
        kalmanFilter_->statePost.at<float>(4) = refBbox.w;
        kalmanFilter_->statePost.at<float>(5) = refBbox.h;

        kalmanFilter_->statePre = kalmanFilter_->statePost;
    }

    bool Update(const cv::Mat &inImg, cv::Rect2d &bboxOut) override
    {
        static time_util::time_point prevTime;
        static time_util::time_point currTime;

        if (!bboxOut)
        {
            cv::Mat meas(config_.measSize, 1, CV_32F);
            meas.at<float>(0) = ballsBox[0].x + ballsBox[0].width / 2;
            meas.at<float>(1) = ballsBox[0].y + ballsBox[0].height / 2;
            meas.at<float>(2) = (float)ballsBox[0].width;
            meas.at<float>(3) = (float)ballsBox[0].height;
        }

        kalmanFilter_->correct();

        return true;
    }

  private:
    std::unique_ptr<cv::KalmanFilter> kalmanFilter_;

    Config config_;
};