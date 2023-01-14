// This script uses the LASOT dataset for initial bounding box extraction and calculating the tracking score.
// Tested with: person, truck, frisbee, basketball, car

// LASOT
// basketball-7
// frisbee-3
// frisbee-7
// truck-7
// truck-9
// truck-16
// truck-17

// VOT-2015
// road

#include "argparse.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

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

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("visual_tracking");
    program.add_argument("--dataset").help("name of the dataset: lasot, vot2015");
    program.add_argument("--image_folder").help("path to the image folder");
    program.parse_args(argc, argv);

    std::string           datasetName{program.get<std::string>("--dataset")};
    std::string           imageFolder;
    std::string           groundTruthFilePath;
    std::vector<cv::Rect> gtBboxVec;

    if (datasetName == "lasot")
    {
        imageFolder         = program.get<std::string>("--image_folder") + ("/img/*.jpg");
        groundTruthFilePath = program.get<std::string>("--image_folder") + ("/groundtruth.txt");
        gtBboxVec           = GetGtBboxFromFileLASOT(groundTruthFilePath);
    }
    else if (datasetName == "vot2015")
    {
        imageFolder         = program.get<std::string>("--image_folder") + ("/*.jpg");
        groundTruthFilePath = program.get<std::string>("--image_folder") + ("/groundtruth.txt");
        gtBboxVec           = GetGtBboxFromFileVOT2015(groundTruthFilePath);
    }
    std::vector<cv::String> imagePaths;
    cv::glob(imageFolder, imagePaths, true);

    // Read the first image to get the bounding box
    cv::Mat  frame{cv::imread(imagePaths.at(0))};
    cv::Rect refBbox{gtBboxVec.at(0)};

    // Create the tracker
    cv::Ptr<cv::Tracker> tracker{cv::TrackerCSRT::create()};
    // cv::Ptr<cv::Tracker> tracker{cv::TrackerKCF::create()};
    tracker->init(frame, refBbox);

    int  frameCtr = 0;
    bool runFree  = false; // run the tracking without user input
    bool reInit   = false; // reinitialize the filter
    for (const auto imagePath : imagePaths)
    {
        frame = cv::imread(imagePath);
        cv::Rect bbox;
        refBbox = gtBboxVec.at(frameCtr);
        cv::rectangle(frame, refBbox, cv::Scalar(255, 0, 0), 2);

        if (reInit)
        {
            tracker->init(frame, refBbox);
        }

        if (tracker->update(frame, bbox))
        {
            cv::rectangle(frame, bbox, cv::Scalar(0, 0, 255), 2);
        }
        else
        {
            cv::putText(frame,
                        "Tracking failure!",
                        cv::Point(100, 80),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.75,
                        cv::Scalar(0, 0, 255),
                        2);
        }

        cv::imshow("frame", frame);
        cv::waitKey(10);
        if (!runFree)
        {
            int inputChar = getchar();
            runFree       = (inputChar == 'c');
            reInit        = (inputChar == 'r') ? true : false;
        }

        frameCtr++;
    }

    return 0;
}
