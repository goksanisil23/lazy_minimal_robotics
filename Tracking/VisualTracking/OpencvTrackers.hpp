#include "Tracker.h"

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

class OpencvTracker : public VisualTracker
{
  public:
    void Init(const cv::Mat &refImg, const cv::Rect &refBbox) override
    {
        tracker_ = cv::TrackerCSRT::create();
        tracker_->init(refImg, refBbox);
    }

    bool Update(const cv::Mat &inImg, cv::Rect2d &bboxOut) override
    {
        cv::Rect bbox;
        if (tracker_->update(inImg, bbox))
        {
            bboxOut = cv::Rect(bbox);
            return true;
        }
        else
        {
            return false;
        }
    }

  private:
    cv::Ptr<cv::Tracker> tracker_;
};