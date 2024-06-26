#pragma once

#include <opencv2/core/core.hpp>

class VisualTracker
{
  public:
    VisualTracker()
    {
    }

    virtual void Init(const cv::Mat &refImg, const cv::Rect &refBbox) = 0;
    virtual bool Update(const cv::Mat &inImg, cv::Rect2d &bboxOut)    = 0;
};
