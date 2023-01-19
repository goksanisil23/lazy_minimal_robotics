#include "Tracker.h"

#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>

class OflowTracker : public VisualTracker
{
  public:
    struct Config
    {
        bool enableGpu{true};
    };

    void Init(const cv::Mat &refImg, const cv::Rect &refBbox) override
    {
        cv::cvtColor(refImg, prevImg_, cv::COLOR_BGR2GRAY);
        prevBbox_ = refBbox;

        if (config_.enableGpu)
        {
            farnGpu_ = cv::cuda::FarnebackOpticalFlow::create();
        }
    }

    bool Update(const cv::Mat &inImg, cv::Rect2d &bboxOut) override
    {
        cv::Mat currImg;
        cv::cvtColor(inImg, currImg, cv::COLOR_BGR2GRAY);
        cv::Mat flow(currImg.size(), CV_32FC2);
        if (!config_.enableGpu)
        {
            cv::calcOpticalFlowFarneback(prevImg_, currImg, flow, 0.5, 3, 15, 10, 5, 1.1, 0);
        }
        else
        {
            cv::cuda::GpuMat prevImgGpu, currImgGpu, flowGpu;
            prevImgGpu.upload(prevImg_);
            currImgGpu.upload(currImg);
            farnGpu_->calc(prevImgGpu, currImgGpu, flowGpu);

            // download the flow field to the host
            flowGpu.download(flow);
        }

        ShowFlowHSV(flow);
        ShowFlowArrows(flow);

        // 1) Calculate the "representative flow" for each bounding box.
        // This is simply the average of all the flow vectors that lie within the bounding box.
        // 2) Warp (shift) the previous ROI based on the current representative flow
        // 3) Association: Find the best match between the estimated ROI and current detector ROI:
        // ---> calculate centroid distance among 2.
        // ---> Calculate IOU among 2.
        // ---> Take average of these 2 as the cost in Hungarian algorithm
        const std::pair<cv::Scalar, cv::Scalar> meanStdDevFlow{GetRepresentativeFlowROI(flow, prevBbox_)};

        // Update the current bounding box
        cv::Rect2d curBbox{prevBbox_.x + meanStdDevFlow.first[0],
                           prevBbox_.y + meanStdDevFlow.first[1],
                           prevBbox_.width,
                           prevBbox_.height};

        // Try various rescaling to see if there is a better bbox that gives less variance
        cv::Rect2d scaledBbox = TryNewScales(flow, curBbox);

        // Update
        bboxOut   = scaledBbox;
        prevBbox_ = scaledBbox;
        prevImg_  = currImg;

        return true;
    }

    cv::Rect2d TryNewScales(const cv::Mat &flow, const cv::Rect2d &curBbox)
    {
        static std::array<double, 8> scales_{0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2};

        auto origMeanStdDevFlow{GetRepresentativeFlowROI(flow, curBbox)};

        double minVariance = origMeanStdDevFlow.second[0] * origMeanStdDevFlow.second[0] +
                             origMeanStdDevFlow.second[1] + origMeanStdDevFlow.second[1];

        cv::Rect2d minVarBbox = curBbox;
        size_t     minIdx     = scales_.size();

        for (size_t i = 0; i < scales_.size(); i++)
        {
            const auto &scale{scales_.at(i)};
            auto        scaledBbox{RescaleBboxFromCenter(curBbox, scale)};
            auto        scaledMeanStdDevFlow{GetRepresentativeFlowROI(flow, scaledBbox)};
            std::cout << "idx: " << i << std::endl;
            std::cout << "mean: " << scaledMeanStdDevFlow.first << std::endl;
            std::cout << "std: " << scaledMeanStdDevFlow.second << std::endl;
            double variance = scaledMeanStdDevFlow.second[0] * scaledMeanStdDevFlow.second[0] +
                              scaledMeanStdDevFlow.second[1] + scaledMeanStdDevFlow.second[1];
            if ((variance < minVariance) && (scaledMeanStdDevFlow.second[0] != 0) &&
                (scaledMeanStdDevFlow.second[1] != 0))
            {
                minVarBbox = scaledBbox;
                minIdx     = i;
            }
        }
        std::cout << "min idx: " << minIdx << std::endl;

        return minVarBbox;
    }

    inline cv::Rect2d RescaleBboxFromCenter(const cv::Rect2d &curBbox, const double &scale)
    {
        double newWidth    = curBbox.width * scale;
        double newHeight   = curBbox.height * scale;
        double newTopLeftX = curBbox.tl().x - (newWidth - curBbox.width) / 2.0;
        double newTopLeftY = curBbox.tl().y - (newHeight - curBbox.height) / 2.0;

        return cv::Rect2d(newTopLeftX, newTopLeftY, newWidth, newHeight);
    }

    std::pair<cv::Scalar, cv::Scalar> GetRepresentativeFlowROI(const cv::Mat &flow, const cv::Rect2d &bbox)
    {
        // // A) NAIVE
        cv::Mat    roiFlow{flow(bbox)};
        cv::Scalar mean, stdDev;
        cv::meanStdDev(roiFlow, mean, stdDev);
        return std::pair<cv::Scalar, cv::Scalar>(mean, stdDev);

        // // B) Remove outliers in flow vector angle
        // cv::Mat flowParts[2];
        // cv::split(flow, flowParts);
        // cv::Mat magnitude, angle, magnitudeNormalized;
        // cv::cartToPolar(flowParts[0], flowParts[1], magnitude, angle, true);

        // cv::Mat anglesRoi{angle(bbox)};
        // cv::Mat magsRoi{magnitude(bbox)};
        // auto    meanAngleRoi{cv::mean(anglesRoi)[0]}; // mean of the angles in the region of interest
        // auto    meanMagRoi{cv::mean(magsRoi)[0]};     // mean of the magnitudes in the region of interest
        // anglesRoi -= meanAngleRoi;
        // magsRoi -= meanMagRoi;

        // cv::Mat patchAng(cv::Mat::zeros(bbox.height, bbox.width, CV_8UC1));
        // cv::Mat patchMag(cv::Mat::zeros(bbox.height, bbox.width, CV_8UC1));

        // // TODO: Do histogram instead of mean?

        // anglesRoi.forEach<float>(
        //     [&patchAng, meanAngleRoi](float &pixel, const int *position)
        //     {
        //         if (pixel < (meanAngleRoi * 0.3)) // look at the angle difference to mean
        //         {
        //             patchAng.at<uint8_t>(position[0], position[1]) = 250;
        //         }
        //     });

        // magsRoi.forEach<float>(
        //     [&patchMag, meanMagRoi](float &pixel, const int *position)
        //     {
        //         if (pixel < (meanMagRoi * 0.3)) // look at the angle difference to mean
        //         {
        //             patchMag.at<uint8_t>(position[0], position[1]) = 125;
        //         }
        //     });

        // cv::imshow("patchAng", patchAng);
        // cv::imshow("patchMag", patchMag);

        // return std::pair<float, float>(mean[0], mean[1]);
        // // return std::pair<float, float>(0, 0);
    }

    void ShowFlowHSV(const cv::Mat &flow)
    {
        // Visualization
        cv::Mat flowParts[2];
        cv::split(flow, flowParts);
        cv::Mat magnitude, angle, magnitudeNormalized;
        cv::cartToPolar(flowParts[0], flowParts[1], magnitude, angle, true);
        cv::normalize(magnitude, magnitudeNormalized, 0.0, 1.0, cv::NORM_MINMAX);
        angle *= ((1.f / 360.f) * (180.f / 255.f));

        //build hsv image
        cv::Mat _hsv[3], hsv, hsv8, bgr;
        _hsv[0] = angle;
        _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
        _hsv[2] = magnitudeNormalized;
        cv::merge(_hsv, 3, hsv);
        hsv.convertTo(hsv8, CV_8U, 255.0);
        cv::cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);

        cv::imshow("flowHSV", bgr);
    }

    void ShowFlowArrows(const cv::Mat &flow)
    {
        cv::Mat flowParts[2];
        cv::split(flow, flowParts);

        static int        xSpace     = 15;
        static int        ySpace     = 15;
        static float      cutoff     = 0.0;
        static float      multiplier = 1.0;
        static cv::Scalar color(0, 255, 0);

        int       x = 0, y = 0;
        float     deltaX = 0.0, deltaY = 0.0, angle = 0.0, mag = 0.0;
        cv::Point p0, p1;

        cv::Mat drawImg;
        cv::cvtColor(prevImg_, drawImg, cv::COLOR_GRAY2BGR);
        // cv::Mat drawImg(flowParts[0].rows, flowParts[0].cols, CV_8UC3, cv::Scalar(0, 0, 0));

        for (y = ySpace; y < flowParts[0].rows; y += ySpace)
        {
            for (x = xSpace; x < flowParts[0].cols; x += xSpace)
            {
                p0.x = x;
                p0.y = y;

                deltaX = (flowParts[0]).at<float>(y, x);
                deltaY = (flowParts[1]).at<float>(y, x);

                angle = atan2(deltaY, deltaX);
                // mag   = sqrt(deltaX * deltaX + deltaY * deltaY);
                mag = 8;

                if (mag > cutoff)
                {
                    p1.x = p0.x + cvRound(multiplier * mag * cos(angle));
                    p1.y = p0.y + cvRound(multiplier * mag * sin(angle));

                    cv::line(drawImg, p0, p1, color, 1, cv::LINE_AA, 0);

                    p0.x = p1.x + cvRound(2 * cos(angle - M_PI + M_PI / 4));
                    p0.y = p1.y + cvRound(2 * sin(angle - M_PI + M_PI / 4));
                    cv::line(drawImg, p0, p1, color, 1, cv::LINE_AA, 0);

                    p0.x = p1.x + cvRound(2 * cos(angle - M_PI - M_PI / 4));
                    p0.y = p1.y + cvRound(2 * sin(angle - M_PI - M_PI / 4));
                    cv::line(drawImg, p0, p1, color, 1, cv::LINE_AA, 0);
                }
            }
        }

        cv::imshow("flow arrows", drawImg);
    }

  private:
    cv::Mat                                 prevImg_;
    cv::Rect2d                              prevBbox_;
    Config                                  config_;
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farnGpu_;
};