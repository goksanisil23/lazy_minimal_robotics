// This script uses the LASOT dataset for initial bounding box extraction and calculating the tracking score.
// Tested with: person, truck, frisbee, basketball, car

#include "argparse.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "Mosse.hpp"

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("visual_tracking");
    program.add_argument("--lasot_folder").help("input image folder from lasot dataset");
    program.parse_args(argc, argv);

    std::string             imageFolder{program.get<std::string>("--lasot_folder") + ("/img/*.jpg")};
    std::vector<cv::String> imagePaths;
    cv::glob(imageFolder, imagePaths, true);

    // Read the first image to get the bounding box
    cv::Mat  frame{cv::imread(imagePaths.at(0))};
    cv::Rect refBbox{cv::selectROI(frame)};

    // Create the tracker
    std::unique_ptr<VisualTracker> tracker{std::make_unique<Mosse>()};
    tracker->Init(frame, refBbox);

    for (const auto imagePath : imagePaths)
    {
        frame = cv::imread(imagePath);
        cv::Rect2d bbox;

        if (tracker->Update(frame, bbox))
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
    }

    return 0;
}