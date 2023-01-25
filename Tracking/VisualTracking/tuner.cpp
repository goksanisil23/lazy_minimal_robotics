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

#include "KalmanTracker.hpp"
#include "KalmanTuner.h"

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("visual_tracking");
    program.add_argument("--dataset").help("path to the dataset folder");
    program.parse_args(argc, argv);

    std::string           datasetFolder = program.get<std::string>("--dataset");
    std::string           imageFolder;
    std::string           groundTruthFilePath;
    std::vector<cv::Rect> gtBboxVec;

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
        std::cerr << "provide dataset for tuning" << std::endl;
        exit(0);
    }

    std::vector<cv::String> imagePaths;
    cv::glob(imageFolder, imagePaths, true);

    std::cout << imagePaths.size() << " images with " << gtBboxVec.size() << " bboxes\n";

    // Create the tracker
    KalmanTuner kalmanTuner;
    kalmanTuner.Tune(gtBboxVec);

    // After tuning is done run the filter
    auto          initState = KalmanTuner::Get6DStateFromBbox(gtBboxVec.at(0));
    KalmanTracker kalmanTracker(initState, kalmanTuner.filterNoiseParams);

    for (size_t i = 1; i < 501; i++)
    {
        const Eigen::Vector<double, 4> gtBboxMes(KalmanTuner::Get4DMeasFromBbox(gtBboxVec.at(i)));
        auto                           predState = kalmanTracker.Predict();
        if (i % 10 == 0)
        {
            kalmanTracker.Correct(gtBboxMes);
        }

        // Show the tracking images
        auto frame    = cv::imread(imagePaths.at(i));
        auto bboxPred = KalmanTuner::GetCvRectFromState(predState);
        cv::rectangle(frame, bboxPred, cv::Scalar(0, 0, 255), 2);
        cv::imshow("frame", frame);
        cv::waitKey(0);
    }

    return 0;
}
