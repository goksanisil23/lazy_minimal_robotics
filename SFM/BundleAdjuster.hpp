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
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3quat.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/src/Core/Matrix.h>

#include "sophus/se3.hpp"

#include <fstream>
#include <iostream>

namespace ba
{

struct Pose6d
{
    Pose6d()
    {
    }

    explicit Pose6d(const Eigen::Vector3d &translationIn, const Eigen::Quaterniond &rotationIn)
    {
        rotation    = Sophus::SO3d(rotationIn);
        translation = translationIn;
    }

    Sophus::SO3d    rotation;
    Eigen::Vector3d translation;
};

// 6 parameters are in the vertex (3 sophus rotation, 3 eigen translation)
class Pose6dVertex : public g2o::BaseVertex<6, Pose6d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override
    {
        _estimate = Pose6d();
    }

    virtual void oplusImpl(const double *update) override
    {
        // _estimate.rotation = Sophus::SO3d(Eigen::Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.rotation = Sophus::SO3d::exp(Eigen::Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Eigen::Vector3d(update[3], update[4], update[5]);
    }

    // Project the 3d point in world coordinates to image
    // input 3d-point in world coordinates is projected by using the known camera intrinsics
    // and the current estimate of the camera pose (extrinsics) within this vertex
    Eigen::Vector2d project3dPointToImage(const Eigen::Vector3d &world_point_3d)
    {
        // landmark point input
        Eigen::Matrix4d P_w(Eigen::Matrix4d::Identity());
        P_w(0, 3) = world_point_3d(0);
        P_w(1, 3) = world_point_3d(1);
        P_w(2, 3) = world_point_3d(2);

        // current camera pose estimate, convert to homogenous transformation
        Eigen::Matrix4d T_w_c(Eigen::Matrix4d::Identity());
        T_w_c(0, 3)             = _estimate.translation(0);
        T_w_c(1, 3)             = _estimate.translation(1);
        T_w_c(2, 3)             = _estimate.translation(2);
        T_w_c.block<3, 3>(0, 0) = _estimate.rotation.unit_quaternion().matrix();
        Eigen::Matrix4d T_c_w   = T_w_c.inverse();

        // 3d Point in camera coordinate frame
        Eigen::Matrix4d P_c(Eigen::Matrix4d::Identity());
        P_c = T_c_w * P_w;

        // project from camera frame to image plane using intrinsics (projection)
        double u = fx_ * -P_c(0, 3) / P_c(2, 3) + cx_;
        double v = fy_ * -P_c(1, 3) / P_c(2, 3) + cy_;

        return Eigen::Vector2d(u, v);
    }

    virtual bool read(std::istream &in)
    {
    }
    virtual bool write(std::ostream &out) const
    {
    }

  private:
    double fx_{512.0};
    double fy_{512.0};
    double cx_{512.0};
    double cy_{320.0};
};

class Point3dVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Point3dVertex()
    {
    }

    virtual void setToOriginImpl() override
    {
        _estimate = Eigen::Vector3d(0, 0, 0);
    }

    virtual void oplusImpl(const double *update) override
    {
        _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(std::istream &in)
    {
    }
    virtual bool write(std::ostream &out) const
    {
    }
};

// Error model template parameters:
// observation (measurement) dimension = 2 -> pixel u & v
// observation type: Vector2d
// connecting vertex types: Pose6dVertex & Point3dVertex
class PerspectiveProjectionEdge : public g2o::BaseBinaryEdge<2, Eigen::Vector2d, Pose6dVertex, Point3dVertex>
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() override
    {
        auto v0                  = static_cast<Pose6dVertex *>(_vertices[0]);
        auto v1                  = static_cast<Point3dVertex *>(_vertices[1]);
        auto projectedPixelCoord = v0->project3dPointToImage(v1->estimate());
        _error                   = projectedPixelCoord - _measurement;
    }

    // clang-format off
    virtual bool read(std::istream &) { return false; }
    virtual bool write(std::ostream &) const { return false; }
    // clang-format on
};

} // namespace ba