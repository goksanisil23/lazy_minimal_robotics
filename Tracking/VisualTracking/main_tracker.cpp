// This script uses the LASOT dataset for initial bounding box extraction and calculating the tracking score.
// Tested with: person, truck, frisbee, basketball, car

// LASOT
// basketball-7
// frisbee-3 /home/goksan/Downloads/tracking_datasets/lasot/frisbee/frisbee-3
// frisbee-7 /home/goksan/Downloads/tracking_datasets/lasot/frisbee/frisbee-7
// truck-7 /home/goksan/Downloads/tracking_datasets/lasot/truck/truck-7
// truck-9 /home/goksan/Downloads/tracking_datasets/lasot/truck/truck-9
// truck-16: /home/goksan/Downloads/tracking_datasets/lasot/truck/truck-16
// truck-17: /home/goksan/Downloads/tracking_datasets/lasot/truck/truck-17

// VOT-2015
// road: /home/goksan/Downloads/tracking_datasets/vot2015/road

// SLOW traffic
// /home/goksan/Downloads/tracking_datasets/slow_traffic

#include "argparse.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>

#include "DatasetUtils.hpp"
#include "TimeUtil.h"

// #include "FeatureTracker.hpp"
// #include "KalmanTracker.hpp"
#include "Mosse.hpp"
#include "OFlowTracker.hpp"
#include "OpencvTrackers.hpp"
#include "StapleTracker.hpp"

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("visual_tracking");
    program.add_argument("--dataset").help("path to the dataset folder");
    program.add_argument("--tracker").help("type of tracker: csrt, oflow");
    program.parse_args(argc, argv);

    std::string           datasetFolder = program.get<std::string>("--dataset");
    std::string           trackerName{program.get<std::string>("--tracker")};
    std::string           imageFolder;
    std::string           groundTruthFilePath;
    std::vector<cv::Rect> gtBboxVec;
    bool                  manualBbox{false};

    if (datasetFolder.find("lasot") != std::string::npos)
    {
        imageFolder         = datasetFolder + ("/img/*.jpg");
        groundTruthFilePath = datasetFolder + ("/groundtruth.txt");
        gtBboxVec           = GetGtBboxFromFileLASOT(groundTruthFilePath);
    }
    else if (datasetFolder.find("vot2015") != std::string::npos)
    {
        imageFolder         = datasetFolder + "/*.jpg";
        groundTruthFilePath = datasetFolder + ("/groundtruth.txt");
        gtBboxVec           = GetGtBboxFromFileVOT2015(groundTruthFilePath);
    }
    else
    {
        manualBbox  = true;
        imageFolder = datasetFolder;
    }

    std::vector<cv::String> imagePaths;
    cv::glob(imageFolder, imagePaths, true);

    // Create the tracker
    std::shared_ptr<VisualTracker> tracker;
    if (trackerName == "csrt")
        tracker = std::make_shared<OpencvTracker>();
    else if (trackerName == "oflow")
        tracker = std::make_shared<OflowTracker>();
    else if (trackerName == "mosse")
        tracker = std::make_shared<Mosse>();
    else if (trackerName == "staple")
        tracker = std::make_shared<StapleTracker>();
    // else if (trackerName == "kalman")
    // {
    //     tracker = std::make_shared<KalmanTracker>();
    //     // Tuning
    //     std::shared_ptr<KalmanTracker> kfPtr(std::dynamic_pointer_cast<KalmanTracker>(tracker));
    //     kfPtr->Tune(gtBboxVec);
    //     exit(0);
    // }
    else
    {
        std::cerr << "no such tracker name" << std::endl;
        return -1;
    }

    // Read the 1st image to get the bounding box
    cv::Mat  frame{cv::imread(imagePaths.at(0))};
    cv::Rect refBbox;
    if (!manualBbox)
        refBbox = gtBboxVec.at(0);
    else
        refBbox = cv::selectROI(frame);

    // Initialize the tracker with the 1st image and associated bbox
    tracker->Init(frame, refBbox);

    int  frameCtr = 0;
    bool runFree  = false; // run the tracking without user input
    bool reInit   = false; // reinitialize the filter
    bool isTracked{true};
    auto t0 = time_util::chronoNow();
    auto t1 = time_util::chronoNow();
    for (const auto imagePath : imagePaths)
    {
        // reInit = (frameCtr % 10 == 0) ? true : false;
        frame = cv::imread(imagePath);
        cv::Mat    drawImg(frame.clone());
        cv::Rect2d bbox;
        if (!manualBbox)
        {
            refBbox = gtBboxVec.at(frameCtr);
            cv::rectangle(drawImg, refBbox, cv::Scalar(255, 0, 0), 2);
            if (reInit)
            {
                tracker->Init(frame, refBbox);
            }
        }
        t0 = time_util::chronoNow();
        if (frameCtr % 30 == 0)
            bbox = refBbox;
        isTracked = tracker->Update(frame, bbox);
        t1        = time_util::chronoNow();
        time_util::showTimeDuration(t1, t0, "Track update: ");
        if (isTracked)
        {
            cv::rectangle(drawImg, bbox, cv::Scalar(0, 0, 255), 2);
        }
        else
        {
            cv::putText(drawImg,
                        "Tracking failure!",
                        cv::Point(100, 80),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.75,
                        cv::Scalar(0, 0, 255),
                        2);
        }

        cv::imshow("frame", drawImg);
        if (!runFree)
        {
            int inputChar = cv::waitKey(0);
            runFree       = (inputChar == 'c');
            reInit        = (inputChar == 'r') ? true : false;
        }
        else
        {
            cv::waitKey(10);
        }

        frameCtr++;
    }

    return 0;
}
