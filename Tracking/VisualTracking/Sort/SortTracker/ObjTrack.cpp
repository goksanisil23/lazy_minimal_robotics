#include "ObjTrack.h"

ObjTrack::ObjTrack(const size_t &id, const Eigen::VectorXd &bbox) : id_(id)
{
    // Initialize the Kalman filter with the bbox detection
    Eigen::VectorXd stateVec{ConvertMeasToState(bbox)};
    kf_ = std::make_unique<KalmanFilter>(6, 4, stateVec);
    InitKfParameters(*kf_);

    snailTrail_ = FixedSizeQueue<raylib::Vector2>(SNAIL_TRAIL_SIZE);
}

Eigen::VectorXd ObjTrack::ConvertMeasToState(const Eigen::VectorXd &bbox) const
{
    // [c_x, c_y, w, h] to [c_x,c_y,v_x,v_y,w,h]
    Eigen::Vector<double, 6> stateVec{bbox(0), bbox(1), 0, 0, bbox(2), bbox(3)};
    return stateVec;
}

Eigen::VectorXd ObjTrack::GetPredBbox() const
{
    // [c_x, c_y, w, h]
    return Eigen::Vector<double, 4>(kf_->x_hat_(0), kf_->x_hat_(1), kf_->x_hat_(4), kf_->x_hat_(5));
}

void ObjTrack::Predict(const double &dt)
{
    kf_->Predict(dt);
    deadReckonCtr_++;

    // Update the snail-trail
    snailTrail_.push((raylib::Vector2){static_cast<float>(kf_->x_hat_(0)), static_cast<float>(kf_->x_hat_(1))});

    Draw();
}
void ObjTrack::Correct(const Eigen::VectorXd &measurement)
{
    kf_->Correct(measurement);
    deadReckonCtr_ = 0; // reset the dead-reckon counter
}

void ObjTrack::InitKfParameters(KalmanFilter &kf)
{
    const size_t stateSize{6};
    const size_t measSize{4};

    kf.stateTransMtx_       = Eigen::MatrixXd::Identity(stateSize, stateSize);
    kf.stateTransMtx_(0, 2) = 0.1;
    kf.stateTransMtx_(1, 3) = 0.1;

    kf.measMtx_       = Eigen::MatrixXd::Zero(measSize, stateSize);
    kf.measMtx_(0, 0) = 1.0;
    kf.measMtx_(1, 1) = 1.0;
    kf.measMtx_(2, 4) = 1.0;
    kf.measMtx_(3, 5) = 1.0;

    kf.measNoiseMtx_ = Eigen::MatrixXd::Identity(measSize, measSize) * 0.1;
    // kf.measNoiseMtx_ = Eigen::MatrixXd::Identity(measSize, measSize) * 100;

    // Diagonals will be populated by tuned vars
    kf.processNoiseMtx_ = Eigen::MatrixXd::Zero(stateSize, stateSize);
    // kf.processNoiseMtx_(0, 0) = 10.0;
    // kf.processNoiseMtx_(1, 1) = 10.0;
    // kf.processNoiseMtx_(2, 2) = 10.0;
    // kf.processNoiseMtx_(3, 3) = 10.0;
    // kf.processNoiseMtx_(4, 4) = 10.0;
    // kf.processNoiseMtx_(5, 5) = 10.0;
    kf.processNoiseMtx_(0, 0) = 1.0;
    kf.processNoiseMtx_(1, 1) = 1.0;
    kf.processNoiseMtx_(2, 2) = 1.0;
    kf.processNoiseMtx_(3, 3) = 1.0;
    kf.processNoiseMtx_(4, 4) = 1.0;
    kf.processNoiseMtx_(5, 5) = 1.0;

    kf.stateErrorCovMtx_       = Eigen::MatrixXd::Identity(stateSize, stateSize);
    kf.stateErrorCovMtx_(2, 2) = 100.0; // larger uncertainty for initially unknown vx,vy
    kf.stateErrorCovMtx_(3, 3) = 100.0;
}

void ObjTrack::Draw() const
{
    // Draw bounding box
    double              topLeftX  = kf_->x_hat_(0) - kf_->x_hat_(4) / 2.0;
    double              topLeftY  = kf_->x_hat_(1) - kf_->x_hat_(5) / 2.0;
    double              botRightX = kf_->x_hat_(0) + kf_->x_hat_(4) / 2.0;
    double              botRightY = kf_->x_hat_(1) + kf_->x_hat_(5) / 2.0;
    raylib::BoundingBox bboxViz(raylib::Vector3(topLeftX, topLeftY, 0), raylib::Vector3(botRightX, botRightY, 0));
    DrawBoundingBox(bboxViz, YELLOW);

    // Draw the id
    DrawText(std::to_string(id_).c_str(), bboxViz.max.x + 5, bboxViz.max.y + 5, 15, YELLOW);
    DrawSnailTrail();
}

void ObjTrack::DrawSnailTrail() const
{
    for (const auto &pos : snailTrail_)
    {
        DrawCircle(pos.x, pos.y, kf_->x_hat_(4) / 15.0, YELLOW);
    }
}