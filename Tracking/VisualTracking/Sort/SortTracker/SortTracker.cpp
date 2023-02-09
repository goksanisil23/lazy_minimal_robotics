#include "SortTracker.h"

#include <numeric>
#include <unordered_map>

#include "TimeUtil.h"

SortTracker::SortTracker()
{
    hungSolver_.costs()->Reserve(1000, 1000);
}

void SortTracker::Step(const std::vector<Eigen::VectorXd> &detections, const double &dt)
{
    // 1) Iterate the object-tracks via object-model in KF (motion model, appearance model, etc)
    for (auto &objTrack : objTracks_)
    {
        objTrack.Predict(dt);
    }

    // 2) Associate the detections to object-tracks
    std::vector<size_t>                    unassignedDetections;
    std::vector<std::pair<size_t, size_t>> assignedTrackToDetectionMap;
    if (!detections.empty())
    {
        AssignDetectionsToTracks(detections, objTracks_, unassignedDetections, assignedTrackToDetectionMap);
    }

    // 3) Update KF with the detections associated to object-tracks
    for (const auto &pairIdx : assignedTrackToDetectionMap)
    {
        objTracks_.at(pairIdx.first).Correct(detections.at(pairIdx.second));
    }

    // 4) Create new object-tracks for unmatched detections
    for (const auto &detIdx : unassignedDetections)
    {
        objTracks_.emplace_back(newIdForObjTrack_, detections.at(detIdx));
        newIdForObjTrack_++;
    }

    // 5) Manage track deletion
}

// @brief Given detections and the existing object tracks, tries to associate detections to tracks.
// Outputs pairings of matched object-track-ids to detection-ids,
// and the remaining indices of unmatched detections
void SortTracker::AssignDetectionsToTracks(const std::vector<Eigen::VectorXd>     &detections,
                                           const std::vector<ObjTrack>            &objTracks,
                                           std::vector<size_t>                    &unassignedDetections,
                                           std::vector<std::pair<size_t, size_t>> &assignedTrackToDetectionMap)
{
    // If there are no objects tracked, all detections should trigger new tracks
    if (objTracks.empty())
    {
        unassignedDetections.resize(detections.size());
        std::iota(unassignedDetections.begin(), unassignedDetections.end(), 0);
        std::for_each(
            unassignedDetections.begin(), unassignedDetections.end(), [](auto el) { std::cout << el << " "; });
        return;
    }

    // Create cost matrix based on IoU score between obj-detections vs track-predictions
    // costMtx = [num_detections x num_tracks]
    Eigen::MatrixXd costMtx(CreateAssingmentCostMatrix(detections, objTracks));
    hungSolver_.costs()->AssignFromMtx(std::move(costMtx));
    hungSolver_.Minimize(&assignedTrackToDetectionMap);
    ShowAssignments(assignedTrackToDetectionMap);
}

Eigen::MatrixXd SortTracker::CreateAssingmentCostMatrix(const std::vector<Eigen::VectorXd> &detections,
                                                        const std::vector<ObjTrack>        &objTracks) const
{
    Eigen::MatrixXd costMtx(detections.size(), objTracks.size());
    for (size_t row = 0; row < detections.size(); row++)
    {
        for (size_t col = 0; col < objTracks.size(); col++)
        {
            // Negate IoU score since higher overlap = lower cost
            costMtx(row, col) = -IoU(detections.at(row), objTracks.at(col).GetPredBbox());
        }
    }
    std::cout << "costMtx\n" << costMtx << std::endl;
    return costMtx;
}

// bbox = [c_x, c_y, w, h]
double SortTracker::IoU(const Eigen::Vector<double, 4> &detectionBBox, const Eigen::Vector<double, 4> &trackBbox) const
{
    double intersectionX1 = std::max(detectionBBox(0) - detectionBBox(2) / 2.0, trackBbox(0) - trackBbox(2) / 2.0);
    double intersectionY1 = std::max(detectionBBox(1) - detectionBBox(3) / 2.0, trackBbox(1) - trackBbox(3) / 2.0);
    double intersectionX2 = std::min(detectionBBox(0) + detectionBBox(2) / 2.0, trackBbox(0) + trackBbox(2) / 2.0);
    double intersectionY2 = std::min(detectionBBox(1) + detectionBBox(3) / 2.0, trackBbox(1) + trackBbox(3) / 2.0);

    double intersectionH = std::max(intersectionY2 - intersectionY1, 0.0);
    double intersectionW = std::max(intersectionX2 - intersectionX1, 0.0);

    double areaIntersection = intersectionW * intersectionH;

    double areaUnion = detectionBBox(2) * detectionBBox(3) + trackBbox(2) * trackBbox(3) - areaIntersection;

    double iou = areaIntersection / areaUnion;

    return iou;
}

void SortTracker::ShowAssignments(const std::vector<std::pair<size_t, size_t>> &assignments) const
{
    for (const auto &assignment : assignments)
    {
        std::cout << assignment.first << " -> " << assignment.second << std::endl;
    }
}