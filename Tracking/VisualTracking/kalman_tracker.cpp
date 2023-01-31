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
// car-19: /home/goksan/Downloads/tracking_datasets/lasot/car/car-19

// VOT-2015
// road:  /home/goksan/Downloads/tracking_datasets/vot2015/road
// tiger: /home/goksan/Downloads/tracking_datasets/vot2015/tiger

// SLOW traffic
// /home/goksan/Downloads/tracking_datasets/slow_traffic

#include "argparse.hpp"
#include <fstream>
#include <opencv2/opencv.hpp>

#include "DatasetUtils.hpp"
#include "TimeUtil.h"

#include "KalmanTracker.hpp"

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("visual_tracking");
    program.add_argument("--dataset")
        .default_value(std::string("../resources/simulated_ball_lasot"))
        .required()
        .help("path to the dataset folder");

    program.parse_args(argc, argv);

    std::string           datasetFolder = program.get<std::string>("--dataset");
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

    // Read the 1st image to get the bounding box
    cv::Mat  frame{cv::imread(imagePaths.at(0))};
    cv::Rect refBbox;
    if (!manualBbox)
        refBbox = gtBboxVec.at(0);
    else
        refBbox = cv::selectROI(frame);

    // Initialize KF with ground truth
    // std::unique_ptr kalmanTracker{std::make_unique<KalmanTracker>(Get6DStateFromBbox(refBbox))};
    // Initialize KF with random state
    std::unique_ptr kalmanTracker{
        std::make_unique<KalmanTracker>(Eigen::Vector<double, 6>{static_cast<double>(frame.cols / 2),
                                                                 static_cast<double>(frame.rows / 2),
                                                                 0,
                                                                 0,
                                                                 static_cast<double>(refBbox.width / 2),
                                                                 static_cast<double>(refBbox.height / 2)})};

    size_t                   frameCtr = 0;
    bool                     runFree  = false; // run the tracking without user input
    Eigen::Vector<double, 6> bboxVec;
    for (const auto imagePath : imagePaths)
    {
        frame = cv::imread(imagePath);
        cv::Mat    drawImg(frame.clone());
        cv::Rect2d bbox;
        if (!manualBbox)
        {
            refBbox = gtBboxVec.at(frameCtr);
            cv::rectangle(drawImg, refBbox, cv::Scalar(255, 0, 0), 2);
        }

        bboxVec = kalmanTracker->Predict();
        bbox    = GetCvRectFromState(bboxVec); //
        if (frameCtr % CORRECTION_PERIOD == 0)
            bboxVec = kalmanTracker->Correct(Get4DMeasFromBbox(refBbox));
        // bbox = GetCvRectFromState(bboxVec);

        cv::rectangle(drawImg, bbox, cv::Scalar(0, 0, 255), 2);

        cv::imshow("frame", drawImg);
        if (!runFree)
        {
            int inputChar = cv::waitKey(0);
            runFree       = (inputChar == 'c');
        }
        else
        {
            cv::waitKey(10);
        }

        frameCtr++;
    }

    return 0;
}
