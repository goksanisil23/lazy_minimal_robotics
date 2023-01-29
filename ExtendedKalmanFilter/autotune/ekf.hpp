#pragma once

#include <Eigen/Dense>
#include <iostream>

class EKF
{

  public:
    EKF(const Eigen::MatrixXd &P0, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, const float &dt)
        : initialized(false), dt(dt), P_post(P0), P_prio(P0), Q(Q), R(R), nis(0)
    {
    }

    void Init(const Eigen::VectorXd &x0)
    {
        x_hat_prio = x0;
        x_hat_post = x0;

        I = Eigen::MatrixXd::Identity(x0.size(), x0.size());

        initialized = true;
    }

    void Predict(const Eigen::VectorXd &U)
    {
        if (!initialized)
        {
            throw std::runtime_error("Filter is not initialized!");
        }

        // Use the motion model to predict a-priori estimate
        this->x_hat_prio   = this->motion_model(this->x_hat_post, U, this->dt);
        Eigen::MatrixXd jF = this->jacobian_F(this->x_hat_prio, U, this->dt);
        P_prio             = jF * P_post * jF.transpose() + Q;
    }

    void Correct(const Eigen::VectorXd &Z)
    {
        Eigen::MatrixXd jH        = this->jacobian_H(this->x_hat_prio);
        Eigen::VectorXd z_predict = this->observation_model(this->x_hat_prio);

        Eigen::MatrixXd S = jH * P_prio * jH.transpose() + R; // innovation covariance
        Eigen::MatrixXd K = P_prio * jH.transpose() * S.inverse();
        this->x_hat_post  = this->x_hat_prio + K * (Z - z_predict);
        P_post            = (I - K * jH) * P_prio;

        nis = CalcNis(Z, z_predict, S);
    }

    // Return the current state
    Eigen::VectorXd GetState()
    {
        return x_hat_post;
    };

    // Return the current state error covariance
    Eigen::MatrixXd GetP()
    {
        return P_post;
    };

    // // Motion model (user defined)
    std::function<
        Eigen::VectorXd(const Eigen::VectorXd &x_hat_post, const Eigen::VectorXd &u_measured, const float &dt)>
        motion_model;

    // Observation model (user defined)
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> observation_model;

    // Jacobian of the motion model
    std::function<
        Eigen::MatrixXd(const Eigen::VectorXd &x_hat_prio, const Eigen::VectorXd &u_measured, const float &dt)>
        jacobian_F;

    // Jacobian of the observation model
    std::function<Eigen::MatrixXd(const Eigen::VectorXd &x_hat_prio)> jacobian_H;

    // calculate Normalized Innovation Squared, based on predicted and actual measurement
    double CalcNis(const Eigen::VectorXd &Z, const Eigen::VectorXd &Z_pred, const Eigen::MatrixXd &S)
    {
        return (Z - Z_pred).transpose() * S.inverse() * (Z - Z_pred);
    }

    // get the calculated NIS value
    double GetNis() const
    {
        return nis;
    }

    // is the filter initialized?
    bool initialized;

  private:
    // Estimated states
    Eigen::VectorXd x_hat_prio, x_hat_post;

    float dt;

    Eigen::MatrixXd P_post, P_prio, Q, R, I;

    float nis; // normalized innovation squared
};