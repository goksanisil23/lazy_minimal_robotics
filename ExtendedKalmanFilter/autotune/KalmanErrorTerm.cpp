#include "KalmanErrorTerm.h"

EIGEN_MAKE_ALIGNED_OPERATOR_NEW

KalmanErrorTerm::KalmanErrorTerm(const std::vector<Eigen::Vector2d>     &noisyCtrlVec,
                                 const std::vector<Eigen::Vector2d>     &noisyMeasVec,
                                 const std::vector<Eigen::Vector2d>     &trueMeasVec,
                                 const Eigen::Vector<double, STATESIZE> &initState)
    : noisyCtrlVec_(noisyCtrlVec), noisyMeasVec_(noisyMeasVec), trueMeasVec_(trueMeasVec), initState_(initState)
{
}

template <typename T>
Eigen::Vector<T, STATESIZE> KalmanErrorTerm::MotionModel(const Eigen::Vector<T, STATESIZE>     &X,
                                                         const Eigen::Vector<double, CTRLSIZE> &U) const
{
    Eigen::Matrix<double, STATESIZE, STATESIZE> A;
    Eigen::Matrix<T, STATESIZE, CTRLSIZE>       B;
    T                                           heading = X(2);
    A << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    B << DT * ceres::cos(heading), T(0.0), DT * ceres::sin(heading), T(0.0), T(0.0), T(DT), T(1.0), T(0.0);

    return (A * X + B * U);
}

template <typename T>
Eigen::Vector<T, MEASSIZE> KalmanErrorTerm::ObservationModel(const Eigen::Vector<T, STATESIZE> &X) const
{
    Eigen::Matrix<double, MEASSIZE, STATESIZE> H;

    H << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;

    return (H * X);
}

// Jacobian of the motion model
template <typename T>
Eigen::Matrix<T, STATESIZE, STATESIZE> KalmanErrorTerm::JacobianF(const Eigen::Vector<T, STATESIZE>     &X_hat_prio,
                                                                  const Eigen::Vector<double, CTRLSIZE> &U) const
{
    Eigen::Matrix<T, STATESIZE, STATESIZE> F;
    double                                 v_in    = U(0);
    T                                      heading = X_hat_prio(2);

    F << T(1.0), T(0.0), (-DT * v_in * ceres::sin(heading)), (DT * ceres::cos(heading)), T(0.0), T(1.0),
        (DT * v_in * ceres::cos(heading)), (DT * ceres::sin(heading)), T(0.0), T(0.0), T(1.0), T(0.0), T(0.0), T(0.0),
        T(0.0), T(1.0);

    return F;
}

// Jacobian of the measurement model
Eigen::Matrix<double, MEASSIZE, STATESIZE> KalmanErrorTerm::JacobianH() const
{
    // normally jacobian is a function of current aprioristate estimate X,
    // but it happened to be a constant matrix here.

    Eigen::Matrix<double, MEASSIZE, STATESIZE> H;
    H << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;

    return H;
}

template <typename T>
Eigen::Vector<T, STATESIZE> KalmanErrorTerm::Predict(Eigen::Vector<T, STATESIZE>                  &xhat,
                                                     Eigen::Matrix<T, STATESIZE, STATESIZE>       &P,
                                                     const Eigen::Vector<double, CTRLSIZE>        &U,
                                                     const Eigen::Matrix<T, STATESIZE, STATESIZE> &Q) const
{
    // Use the motion model to predict a-priori estimate
    xhat                                      = MotionModel(xhat, U);
    Eigen::Matrix<T, STATESIZE, STATESIZE> jF = JacobianF(xhat, U);
    P                                         = jF * P * jF.transpose() + Q;

    return xhat;
}

template <typename T>
Eigen::Vector<T, STATESIZE> KalmanErrorTerm::Correct(Eigen::Vector<T, STATESIZE>                &xhat,
                                                     Eigen::Matrix<T, STATESIZE, STATESIZE>     &P,
                                                     const Eigen::Vector<double, MEASSIZE>      &Z,
                                                     const Eigen::Matrix<T, MEASSIZE, MEASSIZE> &R) const
{
    Eigen::Matrix<double, MEASSIZE, STATESIZE> jH        = JacobianH();
    Eigen::Vector<T, MEASSIZE>                 z_predict = ObservationModel(xhat);

    Eigen::Matrix<T, MEASSIZE, MEASSIZE>  S = jH * P * jH.transpose() + R; // innovation covariance
    Eigen::Matrix<T, STATESIZE, MEASSIZE> K = P * jH.transpose() * S.inverse();
    xhat                                    = xhat + K * Z - K * z_predict;
    P                                       = P - K * jH * P;

    return xhat;
}

template <typename T>
bool KalmanErrorTerm::operator()(const T *const filterNoiseParamsPtr, T *residualsPtr) const
{

    Eigen::Vector<T, STATESIZE> x_hat_;
    x_hat_(0) = T(0);
    x_hat_(1) = T(0);
    x_hat_(2) = T(0);
    x_hat_(3) = T(0);

    // Measurement Noise
    Eigen::Matrix<T, MEASSIZE, MEASSIZE> R = Eigen::Matrix<T, MEASSIZE, MEASSIZE>::Zero();
    R(0, 0)                                = filterNoiseParamsPtr[0];
    R(1, 1)                                = filterNoiseParamsPtr[1];

    Eigen::Matrix<T, STATESIZE, STATESIZE> Q = Eigen::Matrix<T, STATESIZE, STATESIZE>::Zero();
    Q(0, 0)                                  = filterNoiseParamsPtr[2];
    Q(1, 1)                                  = filterNoiseParamsPtr[3];
    Q(2, 2)                                  = filterNoiseParamsPtr[4];
    Q(3, 3)                                  = filterNoiseParamsPtr[5];

    Eigen::Matrix<T, STATESIZE, STATESIZE> P = Eigen::Matrix<T, STATESIZE, STATESIZE>::Identity();

    for (size_t i = 1; i < 501; i++)
    {
        Eigen::Vector<double, CTRLSIZE> noisyCtrlInput = noisyCtrlVec_.at(i - 1);
        Eigen::Vector<double, MEASSIZE> noisyMeas      = noisyMeasVec_.at(i - 1);
        Eigen::Vector<double, MEASSIZE> trueMeas       = trueMeasVec_.at(i - 1);

        // Execute EKF with the noisy measurements
        x_hat_ = Predict(x_hat_, P, noisyCtrlInput, Q);

        x_hat_ = Correct(x_hat_, P, noisyMeas, R);

        residualsPtr[i - 1] = (x_hat_(0) - trueMeas(0)) * (x_hat_(0) - trueMeas(0)) +
                              (x_hat_(1) - trueMeas(1)) * (x_hat_(1) - trueMeas(1));
    }

    return true;
}

ceres::CostFunction *KalmanErrorTerm::Create(const std::vector<Eigen::Vector2d>     &noisyCtrlVec,
                                             const std::vector<Eigen::Vector2d>     &noisyMeasVec,
                                             const std::vector<Eigen::Vector2d>     &trueMeasVec,
                                             const Eigen::Vector<double, STATESIZE> &initState)
{
    // dimension of residual, dimension of opt_var_1
    return (new ::ceres::AutoDiffCostFunction<KalmanErrorTerm, 500, OPT_PARAMS_SIZE>(
        new KalmanErrorTerm(noisyCtrlVec, noisyMeasVec, trueMeasVec, initState)));
}
