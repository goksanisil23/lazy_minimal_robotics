#include "raylib.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>

constexpr double dt           = 0.1;
constexpr int    screenWidth  = 1400;
constexpr int    screenHeight = 1000;

using namespace Eigen;

// State transition model (assuming constant velocity)
VectorXd stateTransitionModel(const VectorXd &x)
{
    VectorXd x_next(4);
    x_next(0) = x(0) + x(2) * dt; // x position update
    x_next(1) = x(1) + x(3) * dt; // y position update
    x_next(2) = x(2);             // constant velocity in x
    x_next(3) = x(3);             // constant velocity in y
    return x_next;
}

// Process model Jacobian (state transition Jacobian)
MatrixXd processJacobian()
{
    MatrixXd F(4, 4);
    F << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1;
    return F;
}

// Measurement model (radar: range and azimuth)
VectorXd measurementModel(const VectorXd &x)
{
    VectorXd z(2);
    z(0) = std::sqrt(x(0) * x(0) + x(1) * x(1)); // range
    z(1) = std::atan2(x(1), x(0));               // azimuth
    return z;
}

// Jacobian of the measurement model
MatrixXd measurementJacobian(const VectorXd &x)
{
    MatrixXd H(2, 4);
    double   range = std::sqrt(x(0) * x(0) + x(1) * x(1));
    H(0, 0)        = x(0) / range; // ∂r/∂x
    H(0, 1)        = x(1) / range; // ∂r/∂y
    H(0, 2) = H(0, 3) = 0;         // ∂r/∂vx and ∂r/∂vy

    H(1, 0) = -x(1) / (range * range); // ∂φ/∂x
    H(1, 1) = x(0) / (range * range);  // ∂φ/∂y
    H(1, 2) = H(1, 3) = 0;             // ∂φ/∂vx and ∂φ/∂vy

    return H;
}

// EKF update step
void EKFUpdate(VectorXd &x, MatrixXd &P, const VectorXd &z, const MatrixXd &R)
{
    VectorXd z_pred = measurementModel(x);
    MatrixXd H      = measurementJacobian(x);

    VectorXd y = z - z_pred; // measurement residual
    MatrixXd S = H * P * H.transpose() + R;
    MatrixXd K = P * H.transpose() * S.inverse();
    x          = x + K * y;
    P          = (MatrixXd::Identity(4, 4) - K * H) * P;
}

void EKFPredict(VectorXd &x, Eigen::MatrixXd &P, const Eigen::MatrixXd &Q)
{
    MatrixXd F = processJacobian();
    x          = stateTransitionModel(x);
    P          = F * P * F.transpose() + Q;
}

// IEKF update step
void IEKFUpdate(VectorXd &x, MatrixXd &P, const VectorXd &z, const MatrixXd &R)
{
    for (int iter = 0; iter < 10; ++iter)
    {
        VectorXd z_pred = measurementModel(x);
        MatrixXd H      = measurementJacobian(x);

        VectorXd y = z - z_pred; // measurement residual
        MatrixXd S = H * P * H.transpose() + R;
        MatrixXd K = P * H.transpose() * S.inverse();
        x          = x + K * y;
        P          = (MatrixXd::Identity(4, 4) - K * H) * P;

        if (y.norm() < 1e-6)
            break; // Convergence check
    }
}

int main()
{

    float START_X  = screenWidth / 2;
    float START_Y  = screenHeight / 2;
    float START_VX = 2.0;
    float START_VY = -2.0;

    // Initial state and covariance
    VectorXd x_ekf(4);
    x_ekf << START_X + 5., START_Y + 5., START_VX, START_VY; // Initial guess for position and velocity

    VectorXd x_iekf(4);
    x_iekf = x_ekf;

    VectorXd x_no_filter(4);
    x_no_filter = x_ekf;

    Eigen::MatrixXd P_ekf  = Eigen::DiagonalMatrix<double, 4>(Eigen::Vector4d(10., 10., 100., 100.)).toDenseMatrix();
    Eigen::MatrixXd P_iekf = P_ekf;

    // Process and measurement noise
    Eigen::MatrixXd Q =
        0.1 * Eigen::DiagonalMatrix<double, 4>(Eigen::Vector4d(0.1, 0.1, 0.01, 0.01)).toDenseMatrix(); // process noise
    Eigen::MatrixXd R =
        Eigen::DiagonalMatrix<double, 2>(Eigen::Vector2d(2e-6, 1e-2)).toDenseMatrix(); // measurement noise

    // Simulate a trajectory for the object
    VectorXd x_true(4);
    x_true << START_X, START_Y, START_VX, START_VY; // Ground truth state

    InitWindow(screenWidth, screenHeight, "iEKF");
    SetTargetFPS(60);

    std::random_device         rd;
    std::mt19937               gen(rd());
    std::normal_distribution<> d(0, 1);

    size_t k = 0;
    while (!WindowShouldClose())
    {
        // Simulate the true state transition
        VectorXd process_noise = (Q.array().sqrt()).matrix() * Eigen::Vector4d(d(gen), d(gen), d(gen), d(gen));
        std::cout << "process noise: " << process_noise.transpose() << std::endl;
        x_true = stateTransitionModel(x_true) + process_noise;

        // No filter
        x_no_filter = stateTransitionModel(x_no_filter);

        EKFPredict(x_ekf, P_ekf, Q);
        EKFPredict(x_iekf, P_iekf, Q);

        // Measurements come in lower frequency
        if (k % 10 == 0)
        {
            VectorXd measurement_noise = (R.array().sqrt()).matrix() * Eigen::Vector2d(d(gen), d(gen));
            std::cout << "measurement noise: " << measurement_noise.transpose() << std::endl;
            VectorXd z = measurementModel(x_true) + measurement_noise;

            EKFUpdate(x_ekf, P_ekf, z, R);
            IEKFUpdate(x_iekf, P_iekf, z, R);
        }

        // Drawing
        {
            BeginDrawing();
            ClearBackground(BLACK);
            constexpr int kFontSize = 10;
            DrawCircleV({x_true(0), x_true(1)}, 20, RED);
            DrawText("x_true", x_true(0), x_true(1), kFontSize, WHITE);
            DrawCircleV({x_ekf(0), x_ekf(1)}, 20, Fade(RED, 0.5f));
            DrawText("x_ekf", x_ekf(0), x_ekf(1), kFontSize, WHITE);
            DrawCircleV({x_iekf(0), x_iekf(1)}, 20, Fade(GREEN, 0.5f));
            DrawText("x_iekf", x_iekf(0), x_iekf(1), kFontSize, WHITE);
            DrawCircleV({x_no_filter(0), x_no_filter(1)}, 20, BLUE);
            DrawText("deadreckon", x_no_filter(0), x_no_filter(1), kFontSize, WHITE);

            EndDrawing();
        }

        std::cout << "P_ekf:  " << P_ekf.norm() << std::endl;
        std::cout << "P_iekf:  " << P_iekf.norm() << std::endl;
        k++;
    }

    return 0;
}
