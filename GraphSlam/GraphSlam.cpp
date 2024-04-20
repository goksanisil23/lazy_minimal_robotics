#include "GraphSlam.h"

namespace
{
class OdometryConstraint
{
  public:
    OdometryConstraint(double dx, double dy, double dtheta) : dx_(dx), dy_(dy), dtheta_(dtheta)
    {
    }

    template <typename T>
    bool operator()(const T *const prev_pose, const T *const current_pose, T *residuals) const
    {
        residuals[0] = current_pose[0] - (prev_pose[0] + dx_ * cos(prev_pose[2]) - dy_ * sin(prev_pose[2]));
        residuals[1] = current_pose[1] - (prev_pose[1] + dx_ * sin(prev_pose[2]) + dy_ * cos(prev_pose[2]));
        residuals[2] = current_pose[2] - (prev_pose[2] + dtheta_);
        return true;
    }

  private:
    const double dx_, dy_, dtheta_;
};

class LandmarkObservationConstraint
{
  public:
    LandmarkObservationConstraint(double range, double bearing)
        : observed_lm_range_(range), observed_lm_bearing_(bearing)
    {
    }

    template <typename T>
    bool operator()(const T *const agent_pose, const T *const landmark_position, T *residuals) const
    {
        T dx                         = landmark_position[0] - agent_pose[0];
        T dy                         = landmark_position[1] - agent_pose[1];
        T predicted_landmark_range   = dx * dx + dy * dy;
        T predicted_landmark_bearing = atan2(dy, dx) - agent_pose[2];

        residuals[0] = predicted_landmark_range - T(observed_lm_range_ * observed_lm_range_);
        residuals[1] = predicted_landmark_bearing - T(observed_lm_bearing_);
        return true;
    }

  private:
    const double observed_lm_range_, observed_lm_bearing_;
};

} // namespace

GraphSlam::GraphSlam(const double init_global_x, const double init_global_y, const double init_global_heading)
    : agent_pose_idx_{0}
{
    agent_poses_[agent_pose_idx_] = Eigen::Vector3d(init_global_x, init_global_y, init_global_heading);
    current_pose_ids_.push(agent_pose_idx_);
    agent_pose_idx_++;
}

void GraphSlam::processMeasurements(const raylib::Vector2                  &agent_delta_pos,
                                    const float                            &agent_delta_head,
                                    const std::vector<LandmarkMeasurement> &landmark_measurements)
{

    // Remove the oldest pose and associated observations in the graph
    if (current_pose_ids_.size() == kNumPosesCapacity)
    {
        auto const removed_agent_pose_id = current_pose_ids_.front();
        current_pose_ids_.pop();

        agent_poses_.erase(removed_agent_pose_id);
        landmark_observations_.erase(std::remove_if(landmark_observations_.begin(),
                                                    landmark_observations_.end(),
                                                    [removed_agent_pose_id](const LandmarkObservation &lm)
                                                    { return lm.agent_pose_id == removed_agent_pose_id; }),
                                     landmark_observations_.end());

        odometry_observations_.erase(std::remove_if(odometry_observations_.begin(),
                                                    odometry_observations_.end(),
                                                    [removed_agent_pose_id](const OdometryObservation &oo)
                                                    { return oo.agent_pose_id_from == removed_agent_pose_id; }),
                                     odometry_observations_.end());
        std::cout << "Removed agent pose " << removed_agent_pose_id << std::endl;
    }

    auto const      last_pose = agent_poses_[current_pose_ids_.back()];
    Eigen::Vector3d new_pose;
    new_pose(0) = last_pose(0) + agent_delta_pos.x * cos(last_pose(2)) - agent_delta_pos.y * sin(last_pose(2));
    new_pose(1) = last_pose(1) + agent_delta_pos.x * sin(last_pose(2)) + agent_delta_pos.y * cos(last_pose(2));
    new_pose(2) = last_pose(2) + agent_delta_head;

    // Add the new pose to the buffer
    current_pose_ids_.push(agent_pose_idx_);
    agent_poses_[agent_pose_idx_] = new_pose;
    std::cout << "added new pose " << agent_pose_idx_ << " to graph" << std::endl;

    odometry_observations_.push_back(OdometryObservation{
        agent_delta_pos.x, agent_delta_pos.y, agent_delta_head, agent_pose_idx_ - 1, agent_pose_idx_});

    // Landmark measurements from the current timestamp
    for (auto const &lm : landmark_measurements)
    {
        landmark_observations_.push_back(LandmarkObservation{lm.range, lm.bearing, lm.id, agent_pose_idx_});

        // Initialize new landmark position based on the current pose and observation
        if (landmarks_.find(lm.id) == landmarks_.end())
        {
            double landmark_global_x = new_pose(0) + lm.range * std::cos(new_pose(2) + lm.bearing);
            double landmark_global_y = new_pose(1) + lm.range * std::sin(new_pose(2) + lm.bearing);

            landmarks_[lm.id] = Eigen::Vector2d(landmark_global_x, landmark_global_y);
            std::cout << "added new landmark " << lm.id << std::endl;
        }
    }

    agent_pose_idx_++;

    runOptimization();
}

void GraphSlam::runOptimization()
{
    ceres::Problem ceres_problem;

    for (const auto &obs : landmark_observations_)
    {
        // 2: dimension of residual
        // 3: 1st parameter block: dimension of robot pose (x,y,theta)
        // 2: 2nd parameter block: dimension of landmark position (x,y)
        ceres::CostFunction *landmark_cost = new ceres::AutoDiffCostFunction<LandmarkObservationConstraint, 2, 3, 2>(
            new LandmarkObservationConstraint(obs.range, obs.bearing));

        ceres_problem.AddResidualBlock(
            landmark_cost, nullptr, agent_poses_[obs.agent_pose_id].data(), landmarks_[obs.landmark_id].data());
    }

    for (const auto &obs : odometry_observations_)
    {
        // 3: dimension of residual
        // 3: 1st parameter block: dimension of robot pose k-1 (x,y,theta)
        // 3: 2nd parameter block: dimension of robot pose k (x,y,theta)
        ceres::CostFunction *odom_cost = new ceres::AutoDiffCostFunction<OdometryConstraint, 3, 3, 3>(
            new OdometryConstraint(obs.dx, obs.dy, obs.dtheta));

        ceres_problem.AddResidualBlock(
            odom_cost, nullptr, agent_poses_[obs.agent_pose_id_from].data(), agent_poses_[obs.agent_pose_id_to].data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type           = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &ceres_problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    most_recent_optimized_agent_pose_ = agent_poses_[agent_pose_idx_ - 1];
}

raylib::Vector2 GraphSlam::getLastOptPose()
{
    return raylib::Vector2(most_recent_optimized_agent_pose_(0), most_recent_optimized_agent_pose_(1));
    // agent_heading  = most_recent_optimized_agent_pose_(2);
}