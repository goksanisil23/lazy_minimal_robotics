#pragma once

#include <vector>

#include "HungarianOptimizer.hpp"
#include "ObjTrack.h"

class SortTracker
{
  public:
    SortTracker();

    void Step(const std::vector<Eigen::VectorXd> &detections, const double &dt);
    void AssignDetectionsToTracks(const std::vector<Eigen::VectorXd>     &detections,
                                  const std::vector<ObjTrack>            &objTracks,
                                  std::vector<size_t>                    &unassignedDetections,
                                  std::vector<std::pair<size_t, size_t>> &assignedTrackToDetectionMap);

    double IoU(const Eigen::Vector<double, 4> &detectionBBox, const Eigen::Vector<double, 4> &trackBbox) const;

    Eigen::MatrixXd CreateAssingmentCostMatrix(const std::vector<Eigen::VectorXd> &detections,
                                               const std::vector<ObjTrack>        &objTracks) const;

    void ShowAssignments(const std::vector<std::pair<size_t, size_t>> &assignments) const;

  private:
    size_t                newIdForObjTrack_{0};
    std::vector<ObjTrack> objTracks_;

    HungarianOptimizer<double> hungSolver_;
};