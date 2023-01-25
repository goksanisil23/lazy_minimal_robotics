#include "KalmanTuner.h"
#include "KalmanErrorTerm.h"

KalmanTuner::KalmanTuner(const Eigen::VectorXd &x0)
{
    x_hat_ = x0;
}

void KalmanTuner::Tune(const std::vector<cv::Rect> &gtBboxes)
{
    ceres::Problem       optProblem;
    ceres::LossFunction *lossFunc{nullptr};

    // Optimization parameters (Process noise & Measurement noise)
    std::vector<double> filterNoiseParams(6, 0.5);
    // std::vector<double> initState(6);
    Eigen::Vector<double, 6> initState(GetStateFromBbox(gtBboxes.at(0)));

    std::vector<Eigen::Vector<double, 4>> gtBboxVec;
    for (size_t i = 1; i < gtBboxes.size(); i++)
    {
        const cv::Rect2d gtBbox(gtBboxes.at(i));
        gtBboxVec.push_back(GetMeasFromBbox(gtBbox));
    }

    ceres::CostFunction *costFunc;
    costFunc = KalmanErrorTerm::Create(gtBboxVec, initState);
    optProblem.AddResidualBlock(costFunc, lossFunc, filterNoiseParams.data());

    ceres::Solver::Options solverOpts;
    solverOpts.linear_solver_type           = ceres::DENSE_QR;
    solverOpts.minimizer_progress_to_stdout = true;
    solverOpts.max_num_iterations           = 100;
    ceres::Solver::Summary solverSummary;

    ceres::Solve(solverOpts, &optProblem, &solverSummary);
    std::cout << solverSummary.FullReport() << "\n";
    std::cout << "Final Noise params:\n";
    std::for_each(filterNoiseParams.begin(), filterNoiseParams.end(), [](double param) { std::cout << param << " "; });
}

// returns c_x,c_y,v_x,v_y,w,h
Eigen::VectorXd KalmanTuner::GetStateFromBbox(const cv::Rect2d &bbox)
{
    Eigen::VectorXd state{Eigen::VectorXd::Zero(stateSize_)};
    state(0) = bbox.tl().x + bbox.width / 2.0;
    state(1) = bbox.tl().y + bbox.height / 2.0;
    state(4) = bbox.width;
    state(5) = bbox.height;

    return state;
}

// returns center_x, center_y, width, height
Eigen::VectorXd KalmanTuner::GetMeasFromBbox(const cv::Rect2d &bbox) const
{
    Eigen::VectorXd meas{Eigen::VectorXd::Zero(measSize_)};
    meas(0) = bbox.tl().x + bbox.width / 2.0;
    meas(1) = bbox.tl().y + bbox.height / 2.0;
    meas(2) = bbox.width;
    meas(3) = bbox.height;

    return meas;
}
