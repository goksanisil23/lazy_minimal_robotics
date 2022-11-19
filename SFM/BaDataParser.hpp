#include <fstream>
#include <iostream>
#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/src/Core/Matrix.h>

namespace ba
{
namespace dataset_parser
{

struct RoboticsPose
{
    explicit RoboticsPose(const float &xIn,
                          const float &yIn,
                          const float &zIn,
                          const float &qxIn,
                          const float &qyIn,
                          const float &qzIn,
                          const float &qwIn)
        : x{xIn}, y{yIn}, z{zIn}, qx{qxIn}, qy{qyIn}, qz{qzIn}, qw{qwIn}
    {
    }

    float x, y, z;
    float qx, qy, qz, qw;
};

struct Observation
{
    explicit Observation(const int    &cameraIdx_in,
                         const int    &landmarkIdx_in,
                         const double &keypointU_in,
                         const double &keypointV_in)
        : cameraIdx{cameraIdx_in}, landmarkIdx{landmarkIdx_in}, keypointU{keypointU_in}, keypointV{keypointV_in}
    {
    }
    int    cameraIdx, landmarkIdx;
    double keypointU, keypointV;
};

void parseBaDataset(const std::string            &baDatasetPath,
                    std::vector<Observation>     &measurements,
                    std::vector<RoboticsPose>    &cameraPoses,
                    std::vector<Eigen::Vector3d> &landmarkPositions)
{
    // First 2 lines are header:
    // num_cameras num_3d_landmarks num_observations
    // 10 300 1084

    std::ifstream baFile(baDatasetPath);
    std::string   line;
    std::getline(baFile, line); // skip 1st line
    std::getline(baFile, line); // get amounts of stuff from 2nd line
    std::istringstream lineStream(line);

    size_t numCameras, numLandmarks, numMeasurements;
    lineStream >> numCameras >> numLandmarks >> numMeasurements;

    // Parse observations
    int line_ctr = 1;
    while (line_ctr <= numMeasurements)
    {
        std::getline(baFile, line);
        std::istringstream istream(line);

        int    cameraIdx, landmarkIdx;
        double keypointU, keypointV;

        istream >> cameraIdx >> landmarkIdx >> keypointU >> keypointV;

        measurements.push_back(Observation(cameraIdx, landmarkIdx, keypointU, keypointV));

        line_ctr++;
    }

    // Parse camera poses
    line_ctr = 1;
    while (line_ctr <= numCameras)
    {
        std::getline(baFile, line);
        std::istringstream istream(line);

        float x, y, z, qx, qy, qz, qw;
        istream >> x >> y >> z >> qx >> qy >> qz >> qw;

        cameraPoses.push_back(RoboticsPose(x, y, z, qx, qy, qz, qw));

        line_ctr++;
    }

    // Parse landmark positions
    line_ctr = 1;
    while (line_ctr <= numLandmarks)
    {
        std::getline(baFile, line);
        std::istringstream istream(line);

        float x, y, z;
        istream >> x >> y >> z;

        landmarkPositions.push_back(Eigen::Vector3d(x, y, z));

        line_ctr++;
    }
}

} // namespace dataset_parser
} // namespace ba
