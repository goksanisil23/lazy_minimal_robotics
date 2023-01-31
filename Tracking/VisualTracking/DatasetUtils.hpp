#include <fstream>
#include <iostream>

#include "Eigen/Core"
#include "Eigen/Dense"

#include <opencv2/opencv.hpp>

cv::Rect getAxisAlignedBB(std::vector<cv::Point2f> polygon)
{
    double   cx = double(polygon[0].x + polygon[1].x + polygon[2].x + polygon[3].x) / 4.;
    double   cy = double(polygon[0].y + polygon[1].y + polygon[2].y + polygon[3].y) / 4.;
    double   x1 = std::min(std::min(std::min(polygon[0].x, polygon[1].x), polygon[2].x), polygon[3].x);
    double   x2 = std::max(std::max(std::max(polygon[0].x, polygon[1].x), polygon[2].x), polygon[3].x);
    double   y1 = std::min(std::min(std::min(polygon[0].y, polygon[1].y), polygon[2].y), polygon[3].y);
    double   y2 = std::max(std::max(std::max(polygon[0].y, polygon[1].y), polygon[2].y), polygon[3].y);
    double   A1 = norm(polygon[1] - polygon[2]) * norm(polygon[2] - polygon[3]);
    double   A2 = (x2 - x1) * (y2 - y1);
    double   s  = sqrt(A1 / A2);
    double   w  = s * (x2 - x1) + 1;
    double   h  = s * (y2 - y1) + 1;
    cv::Rect rect(std::round(cx - 1 - w / 2.0), std::round(cy - 1 - h / 2.0), std::round(w), std::round(h));
    return rect;
}

std::vector<cv::Rect> GetGtBboxFromFileVOT2015(const std::string &gtFilePath)
{
    std::vector<cv::Rect> gtBboxVec;
    std::string           gtLine;
    std::ifstream         gtFile(gtFilePath);
    float                 x1, y1, x2, y2, x3, y3, x4, y4;
    while (std::getline(gtFile, gtLine))
    {
        std::replace(gtLine.begin(), gtLine.end(), ',', ' ');
        std::stringstream ss(gtLine);
        ss >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4;
        std::vector<cv::Point2f> polygon;
        polygon.push_back(cv::Point2f(x1, y1));
        polygon.push_back(cv::Point2f(x2, y2));
        polygon.push_back(cv::Point2f(x3, y3));
        polygon.push_back(cv::Point2f(x4, y4));
        gtBboxVec.emplace_back(getAxisAlignedBB(polygon));
    }
    return gtBboxVec;
}

std::vector<cv::Rect> GetGtBboxFromFileLASOT(const std::string &gtFilePath)
{
    std::vector<cv::Rect> gtBboxVec;
    std::string           gtLine;
    std::ifstream         gtFile(gtFilePath);
    while (std::getline(gtFile, gtLine))
    {
        std::stringstream ss(gtLine);
        int               bboxVal[4];
        int               idx = 0;
        int               val;
        while (ss >> val)
        {
            bboxVal[idx] = val;
            if (ss.peek() == ',')
            {
                ss.ignore();
            }
            idx++;
        }
        gtBboxVec.emplace_back(cv::Rect{bboxVal[0], bboxVal[1], bboxVal[2], bboxVal[3]});
    }
    return gtBboxVec;
}

// returns c_x,c_y,v_x,v_y,w,h
Eigen::VectorXd Get6DStateFromBbox(const cv::Rect2d &bbox)
{
    Eigen::VectorXd state{Eigen::VectorXd::Zero(6)};
    state(0) = bbox.tl().x + bbox.width / 2.0;
    state(1) = bbox.tl().y + bbox.height / 2.0;
    state(4) = bbox.width;
    state(5) = bbox.height;

    return state;
}

// returns center_x, center_y, width, height
Eigen::VectorXd Get4DMeasFromBbox(const cv::Rect2d &bbox)
{
    Eigen::VectorXd meas{Eigen::VectorXd::Zero(4)};
    meas(0) = bbox.tl().x + bbox.width / 2.0;
    meas(1) = bbox.tl().y + bbox.height / 2.0;
    meas(2) = bbox.width;
    meas(3) = bbox.height;

    return meas;
}

cv::Rect GetCvRectFromState(const Eigen::Vector<double, 6> &state)
{
    cv::Point topLeft(state(0) - state(4) / 2.0, state(1) - state(5) / 2.0);
    cv::Point botRight(state(0) + state(4) / 2.0, state(1) + state(5) / 2.0);
    cv::Rect  rect(topLeft, botRight);
    return rect;
}