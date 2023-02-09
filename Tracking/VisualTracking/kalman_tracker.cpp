// This script uses a Kalman filter with a linear velocity motion model to do
// single object tracking. The detections come every CORRECTION_PERIOD interval
// In between, Kalman filter basically deadreckons from the last state.
// Tested with the datasets shown below.

// --- Kalman filter --- //
// state = [c_x,c_y,v_x,v_y,w,h] --> (center of bbox, velocity of bbox, height, width)
// measurement = [c_x, c_y, w, h]

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

#include "KalmanFilter.hpp"

constexpr double DT{0.05};
constexpr int    CORRECTION_PERIOD{5};

void initializeKalmanParameters(KalmanFilter &kf)
{
    const size_t stateSize{6};
    const size_t measSize{4};

    kf.stateTransMtx_       = Eigen::MatrixXd::Identity(stateSize, stateSize);
    kf.stateTransMtx_(0, 2) = 0.1;
    kf.stateTransMtx_(1, 3) = 0.1;

    kf.measMtx_       = Eigen::MatrixXd::Zero(measSize, stateSize);
    kf.measMtx_(0, 0) = 1.0;
    kf.measMtx_(1, 1) = 1.0;
    kf.measMtx_(2, 4) = 1.0;
    kf.measMtx_(3, 5) = 1.0;

    kf.measNoiseMtx_ = Eigen::MatrixXd::Identity(measSize, measSize) * 100;
    // measNoiseMtx_ = Eigen::MatrixXd::Identity(measSize, measSize) * 1000;

    // Diagonals will be populated by tuned vars
    kf.processNoiseMtx_       = Eigen::MatrixXd::Zero(stateSize, stateSize);
    kf.processNoiseMtx_(0, 0) = 10.0;
    kf.processNoiseMtx_(1, 1) = 10.0;
    kf.processNoiseMtx_(2, 2) = 10.0;
    kf.processNoiseMtx_(3, 3) = 10.0;
    kf.processNoiseMtx_(4, 4) = 10.0;
    kf.processNoiseMtx_(5, 5) = 10.0;

    kf.stateErrorCovMtx_       = Eigen::MatrixXd::Identity(stateSize, stateSize);
    kf.stateErrorCovMtx_(2, 2) = 100.0; // larger uncertainty for initially unknown vx,vy
    kf.stateErrorCovMtx_(3, 3) = 100.0;
}

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

    // --- Kalman filter --- //
    // state = [c_x,c_y,v_x,v_y,w,h] --> (center of bbox, velocity of bbox, height, width)
    // measurement = [c_x, c_y, w, h]

    // 1) Initialize KF with ground truth
    // auto initState = Get6DStateFromBbox(refBbox);
    // 2) Initialize KF with random state
    auto initState = Eigen::Vector<double, 6>{static_cast<double>(frame.cols / 2),
                                              static_cast<double>(frame.rows / 2),
                                              0,
                                              0,
                                              static_cast<double>(refBbox.width / 2),
                                              static_cast<double>(refBbox.height / 2)};

    std::unique_ptr kalmanTracker{std::make_unique<KalmanFilter>(6, 4, initState)};
    initializeKalmanParameters(*kalmanTracker);

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

        bboxVec = kalmanTracker->Predict(DT);
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