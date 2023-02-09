#pragma once

#include <vector>

#include "HungarianOptimizer.hpp"
#include "ObjTrack.h"

class SortTracker
{
  public:
    // # cycles allowed to deadreckon before deleting the track, until a detection comes
    static constexpr size_t DEADRECKON_CTR_LIMIT = 5;
    static constexpr double IOU_THRESHOLD        = 0.3;

  public:
    SortTracker();

    void Step(const std::vector<Eigen::VectorXd> &detections, const double &dt);
    void AssignDetectionsToTracks(const std::vector<Eigen::VectorXd>     &detections,
                                  const std::vector<ObjTrack>            &objTracks,
                                  std::vector<size_t>                    &unassignedDetIdxs,
                                  std::vector<std::pair<size_t, size_t>> &assignedDetectionToTrackMap);

    double IoU(const Eigen::Vector<double, 4> &detectionBBox, const Eigen::Vector<double, 4> &trackBbox) const;

    Eigen::MatrixXd CreateAssingmentCostMatrix(const std::vector<Eigen::VectorXd> &detections,
                                               const std::vector<ObjTrack>        &objTracks) const;

    void ShowAssignments(const std::vector<std::pair<size_t, size_t>> &assignments) const;
    void ShowDetections(const std::vector<Eigen::VectorXd> &detections) const;

  private:
    size_t                newIdForObjTrack_{0};
    std::vector<ObjTrack> objTracks_;

    HungarianOptimizer<double> hungSolver_;
};