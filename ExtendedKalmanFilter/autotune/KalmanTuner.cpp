#include "KalmanTuner.h"
#include "KalmanErrorTerm.h"

KalmanTuner::KalmanTuner() : filterNoiseParams(OPT_PARAMS_SIZE, INIT_OPT_PARAMS_VAL)
{
}

void KalmanTuner::Tune(const std::vector<Eigen::Vector2d> &noisyCtrlVec,
                       const std::vector<Eigen::Vector2d> &noisyMeasVec,
                       const std::vector<Eigen::Vector2d> &trueMeasVec)
{
    ceres::Problem       optProblem;
    ceres::LossFunction *lossFunc{nullptr};

    // Optimization parameters (Process noise & Measurement noise)
    Eigen::Vector<double, STATESIZE> initState{0, 0, 0, 0};

    ceres::CostFunction *costFunc;
    costFunc = KalmanErrorTerm::Create(noisyCtrlVec, noisyMeasVec, trueMeasVec, initState);
    optProblem.AddResidualBlock(costFunc, lossFunc, filterNoiseParams.data());

    ceres::Solver::Options solverOpts;
    solverOpts.linear_solver_type           = ceres::DENSE_QR;
    solverOpts.minimizer_progress_to_stdout = true;
    solverOpts.max_num_iterations           = 500;
    ceres::Solver::Summary solverSummary;

    ceres::Solve(solverOpts, &optProblem, &solverSummary);
    std::cout << solverSummary.FullReport() << "\n";
    std::cout << "Final Noise params:\n";
    std::for_each(filterNoiseParams.begin(), filterNoiseParams.end(), [](double param) { std::cout << param << " "; });
    std::cout << std::endl;
}
