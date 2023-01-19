#pragma once

#include "Tracker.h"

#include "staple_tracker.hpp"

class StapleTracker : public VisualTracker
{
  public:
    void Init(const cv::Mat &refImg, const cv::Rect &refBbox) override
    {
        staple_.tracker_staple_initialize(refImg, refBbox);
        staple_.tracker_staple_train(refImg, true);
    }

    bool Update(const cv::Mat &inImg, cv::Rect2d &bboxOut) override
    {
        bboxOut = staple_.tracker_staple_update(inImg);
        staple_.tracker_staple_train(inImg, false);

        return true;
    }

  private:
    STAPLE_TRACKER staple_;
};