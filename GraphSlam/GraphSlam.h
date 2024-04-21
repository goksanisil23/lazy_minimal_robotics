#include "raylib-cpp.hpp"
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <queue>
#include <unordered_map>

#include "Landmark.h"

struct LandmarkObservation
{
    double range;
    double bearing;
    int    landmark_id;
    size_t agent_pose_id;
};

struct OdometryObservation
{
    double dx, dy, dtheta;
    size_t agent_pose_id_from;
    size_t agent_pose_id_to;
};

class GraphSlam
{
  public:
    // Number of odometry poses needed to run the graph optimization
    static constexpr size_t kNumPosesNeeded{20};
    // minimum distance needed between odometry pose to be added to the graph
    static constexpr float kDeltaMotionLimit{10.0};
    // Total number of poses to be kept in the sliding window
    static constexpr size_t kNumPosesCapacity{20};

    GraphSlam(const double init_global_x, const double init_global_y, const double init_global_heading);

    void processMeasurements(const raylib::Vector2                  &agent_delta_pos,
                             const float                            &agent_delta_head,
                             const std::vector<LandmarkMeasurement> &landmark_measurements);

    Eigen::Vector3d getLastOptPose();

  private:
    void runOptimization();

  public:
    // Parameters to be optimized
    // NOTE: Ceres requires parameters to be optimized to be double
    std::unordered_map<size_t, Eigen::Vector3d> agent_poses_{}; // x,y,theta, mapping from agent pose idx
    std::unordered_map<int, Eigen::Vector2d>    landmarks_;     // x,y, mapping from landmark id

    std::queue<size_t> current_pose_ids_{};
    size_t             agent_pose_idx_{0};

    std::vector<LandmarkObservation> landmark_observations_{};
    std::vector<OdometryObservation> odometry_observations_{};

    raylib::Vector2 pose_accumulator_{raylib::Vector2{0, 0}};

    Eigen::Vector3d most_recent_optimized_agent_pose_;
};