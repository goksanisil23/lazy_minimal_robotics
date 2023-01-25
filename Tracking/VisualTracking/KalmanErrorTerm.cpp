#include "KalmanErrorTerm.h"

EIGEN_MAKE_ALIGNED_OPERATOR_NEW

KalmanErrorTerm::KalmanErrorTerm(const std::vector<Eigen::Vector<double, 4>> &gtBboxVec,
                                 const Eigen::Vector<double, 6>              &initState)
    : gtBboxVec_(gtBboxVec), initState_(initState)
{
}

template <typename T>
bool KalmanErrorTerm::operator()(const T *const filterNoiseParamsPtr, T *residualsPtr) const
{
    // Create Kalman Filter
    const int32_t stateSize_{6}; // c_x,c_y,v_x,v_y,w,h (center of bbox, velocity of bbox, height, width)
    const int32_t measSize_{4};  // c_x, c_y, w, h

    // Constants
    const double                                  dt{0.05};
    Eigen::Matrix<double, stateSize_, stateSize_> stateTransMtx_ = Eigen::MatrixXd::Identity(stateSize_, stateSize_);
    stateTransMtx_(0, 2)                                         = dt;
    stateTransMtx_(1, 3)                                         = dt;
    Eigen::Matrix<double, measSize_, stateSize_> measMtx_        = Eigen::MatrixXd::Zero(measSize_, stateSize_);
    measMtx_(0, 0)                                               = 1.0;
    measMtx_(1, 1)                                               = 1.0;
    measMtx_(2, 4)                                               = 1.0;
    measMtx_(3, 5)                                               = 1.0;
    Eigen::Matrix<double, measSize_, measSize_> measNoiseMtx_ = Eigen::MatrixXd::Identity(measSize_, measSize_) * 1e-7;

    // Templated vars
    Eigen::Vector<T, stateSize_> x_hat_;
    x_hat_(0) = T(initState_(0));
    x_hat_(1) = T(initState_(1));
    x_hat_(2) = T(initState_(2));
    x_hat_(3) = T(initState_(3));
    x_hat_(4) = T(initState_(4));
    x_hat_(5) = T(initState_(5));
    // Diagonals will be populated by tuning vars
    Eigen::Matrix<T, stateSize_, stateSize_> processNoiseMtx_  = Eigen::Matrix<T, stateSize_, stateSize_>::Zero();
    Eigen::Matrix<T, stateSize_, stateSize_> stateErrorCovMtx_ = Eigen::Matrix<T, stateSize_, stateSize_>::Identity();

    // Formulate the error of the Kalman filter's prediction based on current tunable parameters

    // // Update the tuning parameters
    processNoiseMtx_(0, 0) = filterNoiseParamsPtr[0];
    processNoiseMtx_(1, 1) = filterNoiseParamsPtr[1];
    processNoiseMtx_(2, 2) = filterNoiseParamsPtr[2];
    processNoiseMtx_(3, 3) = filterNoiseParamsPtr[3];
    processNoiseMtx_(4, 4) = filterNoiseParamsPtr[4];
    processNoiseMtx_(5, 5) = filterNoiseParamsPtr[5];

    for (size_t i = 1; i < 501; i++)
    {
        const Eigen::Vector<double, 4> gtBboxMes(gtBboxVec_.at(i));

        // Predict
        x_hat_            = stateTransMtx_ * x_hat_;
        stateErrorCovMtx_ = (stateTransMtx_ * stateErrorCovMtx_ * stateTransMtx_.transpose()) + processNoiseMtx_;

        // Update residual before correction
        Eigen::Vector<T, stateSize_> x_hat(x_hat_);
        residualsPtr[i - 1] = IoU(gtBboxMes, x_hat);

        // Run measurement update every 10th frame
        if (i % 10 == 0)
        {
            Eigen::Matrix<T, stateSize_, measSize_> K =
                stateErrorCovMtx_ * measMtx_.transpose() *
                (measMtx_ * stateErrorCovMtx_ * measMtx_.transpose() + measNoiseMtx_).inverse();

            x_hat_            = x_hat_ + K * gtBboxMes - K * measMtx_ * x_hat_;
            stateErrorCovMtx_ = stateErrorCovMtx_ - K * measMtx_ * stateErrorCovMtx_;
        }
    }

    return true;
}

ceres::CostFunction *KalmanErrorTerm::Create(const std::vector<Eigen::Vector<double, 4>> &gtBboxVec,
                                             const Eigen::Vector<double, 6>              &initState)
{
    // dimension of residual, dimension of opt_var_1
    return (new ::ceres::AutoDiffCostFunction<KalmanErrorTerm, 500, 6>(new KalmanErrorTerm(gtBboxVec, initState)));
}

// c_x,c_y,w,h
template <typename T>
T KalmanErrorTerm::IoU(const Eigen::Vector<double, 4> &gtBboxMes, const Eigen::Vector<T, 6> &predState) const
{

    T intersectionX1 = std::min(T(gtBboxMes(0) - gtBboxMes(2) / 2.0), predState(0) - predState(4) / 2.0);
    T intersectionY1 = std::min(T(gtBboxMes(1) - gtBboxMes(3) / 2.0), predState(1) - predState(5) / 2.0);
    T intersectionX2 = std::min(T(gtBboxMes(0) + gtBboxMes(2) / 2.0), predState(0) + predState(4) / 2.0);
    T intersectionY2 = std::min(T(gtBboxMes(1) + gtBboxMes(3) / 2.0), predState(1) + predState(5) / 2.0);

    T intersectionH = std::max(intersectionY2 - intersectionY1, T(0.0));
    T intersectionW = std::max(intersectionX2 - intersectionX1, T(0.0));

    T areaIntersection = intersectionW * intersectionH;

    T areaUnion = gtBboxMes(2) * gtBboxMes(3) + predState(4) * predState(5) - areaIntersection;

    T iou = areaIntersection / areaUnion;
    return iou;
}
