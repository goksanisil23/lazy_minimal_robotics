// This script uses the LASOT dataset for initial bounding box extraction and calculating the tracking score.
// Tested with: person, truck, frisbee, basketball, car

// basketball-7
// frisbee-3
// frisbee-7
// truck-7
// truck-9
// truck-16
// truck-17

#include "argparse.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

std::vector<cv::Rect> GetGtBboxFromFile(const std::string &gtFilePath)
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
    program.add_argument("--lasot_folder").help("input image folder from lasot dataset");
    program.parse_args(argc, argv);

    std::string             imageFolder{program.get<std::string>("--lasot_folder") + ("/img/*.jpg")};
    std::vector<cv::String> imagePaths;
    cv::glob(imageFolder, imagePaths, true);

    std::string           groundTruthFilePath{program.get<std::string>("--lasot_folder") + ("/groundtruth.txt")};
    std::vector<cv::Rect> gtBboxVec{GetGtBboxFromFile(groundTruthFilePath)};

    // Read the first image to get the bounding box
    cv::Mat    frame{cv::imread(imagePaths.at(0))};
    cv::Rect2d refBbox{gtBboxVec.at(0)};

    // Create the tracker
    // cv::Ptr<cv::legacy::TrackerMedianFlow> tracker{cv::legacy::TrackerMedianFlow::create()};
    cv::Ptr<cv::legacy::TrackerMedianFlow> tracker{cv::legacy::TrackerMedianFlow::create()};
    tracker->init(frame, refBbox);

    int frameCtr = 0;
    for (const auto imagePath : imagePaths)
    {
        frame = cv::imread(imagePath);
        cv::Rect2d bbox;
        refBbox = gtBboxVec.at(frameCtr);
        cv::rectangle(frame, refBbox, cv::Scalar(255, 0, 0), 2);

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
        cv::waitKey(0);
        frameCtr++;
    }

    return 0;
}
