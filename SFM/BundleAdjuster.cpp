#include "g2o/core/auto_differentiation.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/sparse_optimizer.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/src/Core/Matrix.h>

#include <fstream>
#include <iostream>

struct ImageAndKeypoint
{
    ImageAndKeypoint(const int &cam_idx, const int &x_coord, const int &y_coord)
        : camIdx{cam_idx}, xCoord{x_coord}, yCoord{y_coord} {}
    int camIdx;
    int xCoord, yCoord;
};

struct Landmark
{
    std::vector<ImageAndKeypoint> imgsAndKps;
    Eigen::Vector3f location; // global coordinates of the landmark
};

void linkOrInsertObservation(std::vector<Landmark> &landmarks, const ImageAndKeypoint &ikp1, const ImageAndKeypoint &ikp2, const Eigen::Vector3f &location3d)
{
    // std::vector<Landmark> landmarks_new = landmarks;
    if (!landmarks.empty())
    {
        bool foundExisting = false;
        for (auto &landmark : landmarks)
        {
            for (auto imgAndKp : landmark.imgsAndKps)
            {
                if ((ikp1.camIdx == imgAndKp.camIdx) && (ikp1.xCoord == imgAndKp.xCoord) && (ikp1.yCoord == imgAndKp.yCoord))
                {
                    landmark.imgsAndKps.push_back(ikp2);
                    foundExisting = true;
                    // goto endloop;
                }
            }
        }
        if (!foundExisting)
        {
            Landmark lm;
            lm.imgsAndKps.push_back(ikp1);
            lm.imgsAndKps.push_back(ikp2);
            lm.location = location3d;
            landmarks.push_back(lm);
        }
    }
    else
    {
        // Insert the 1st landmark
        Landmark lm;
        lm.imgsAndKps.push_back(ikp1);
        lm.imgsAndKps.push_back(ikp2);
        lm.location = location3d;
        landmarks.push_back(lm);
    }

    // endloop:
    return;
}

void getUniqueLandmarks(const std::vector<Landmark> &landmarks, std::vector<Landmark> &uniqueLandmarks)
{
    for (const auto &landmark : landmarks)
    {
        if (landmark.imgsAndKps.size() > 2)
        {
            bool isIdentical = false;
            for (int ii = 0; ii < landmark.imgsAndKps.size() - 1; ii++)
            {
                auto src = landmark.imgsAndKps.at(ii);
                for (int jj = ii + 1; jj < landmark.imgsAndKps.size(); jj++)
                {
                    auto trgt = landmark.imgsAndKps.at(jj);
                    isIdentical = isIdentical || ((src.xCoord == trgt.xCoord) && (src.yCoord == trgt.yCoord)) || (src.camIdx == trgt.camIdx);
                }
            }
            if (!isIdentical)
            {
                uniqueLandmarks.push_back(landmark);
            }
        }
    }
}

// Need to associate unique landmarks to observations
void findGlobalMatches(const std::string &ba_data_path, std::vector<Landmark> &landmarks)
{
    std::fstream baDataStream;
    baDataStream.open(ba_data_path);
    std::string line;
    std::getline(baDataStream, line); // skip 1st header line

    while (std::getline(baDataStream, line))
    {
        std::stringstream stream(line);
        int imgIdx_k_minus, imgIdx_k;
        int xk_minus, yk_minus, xk, yk;
        float pt_x, pt_y, pt_z;
        stream >> imgIdx_k_minus >> xk_minus >> yk_minus >> imgIdx_k >> xk >> yk >> pt_x >> pt_y >> pt_z;
        // std::cout << "----- img: " << imgIdx_k_minus << std::endl;

        ImageAndKeypoint ikp1(imgIdx_k_minus, xk_minus, yk_minus);
        ImageAndKeypoint ikp2(imgIdx_k, xk, yk);
        Eigen::Vector3f location3d(pt_x, pt_y, pt_z);
        linkOrInsertObservation(landmarks, ikp1, ikp2, location3d);
    }
}

int main()
{
    std::vector<Landmark> landmarks, unique_landmarks;
    findGlobalMatches("../resources/data_for_ba_rounded.txt", landmarks);
    getUniqueLandmarks(landmarks, unique_landmarks);

    int lm_ctr = 0;
    for (const auto &landmark : unique_landmarks)
    {
        std::cout << "landmark: " << lm_ctr << std::endl;
        for (auto lm_ikp : landmark.imgsAndKps)
        {
            std::cout << lm_ikp.camIdx << " " << lm_ikp.xCoord << " " << lm_ikp.yCoord << std::endl;
        }
        lm_ctr++;
    }

    // TODO: Draw the matches found in unique_landmarks as a sanity check to see if they indeed match

    return 0;
}