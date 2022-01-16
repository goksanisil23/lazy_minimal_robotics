#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

class UKF {

public: 

    UKF(const Eigen::MatrixXd& P0,
        const Eigen::MatrixXd& Q,
        const Eigen::MatrixXd& R,
        const float& dt,
        const float& alpha,
        const float& beta,
        const float& kappa) : initialized(false), P_post(P0), P_prio(P0), Q(Q), R(R), dt(dt), nis(0),
                              alpha(alpha), beta(beta), kappa(kappa)
    {}


    void init(const Eigen::VectorXd& x0) {
        x_hat_prio = x0;
        x_hat_post = x0;
        n_x = x0.size();
        n_z = R.cols();

        I = Eigen::MatrixXd::Identity(x0.size(), x0.size());

        init_sigma_weights();

        initialized = true;
    }

    void init_sigma_weights() {
        w_a.resize(2 * n_x + 1);
        w_c.resize(2 * n_x + 1);
        
        // Method 1:
        w_a(0) = (alpha*alpha*kappa - n_x) / (alpha*alpha*kappa);
        w_c(0) = w_a(0) + 1.0 - alpha*alpha + beta;

        // Keep track of the sum of the weights
        double sum_w_a = w_a(0);
        double sum_w_c = w_c(0);

        for(int i = 1; i <= (2*n_x); i++) {
            w_a(i) = 1 / (2.0*alpha*alpha*kappa);
            w_c(i) = w_a(i);       
            sum_w_a += w_a(i);
            sum_w_c += w_c(i);
        }
        gamma = alpha * std::sqrt(kappa); // adjusts the spread of sigma points

        // // Method 2:
        // w_a(0) = 0.5;
        // w_c(0) = w_a(0);

        // // // Keep track of the sum of the weights
        // double sum_w_a = w_a(0);
        // double sum_w_c = w_c(0);        
        
        // for(int i = 1; i <= (2*n_x); i++) {
        //     w_a(i) = (1.0 - w_a(0)) / (2*n_x);
        //     w_c(i) = w_a(i);       
        //     sum_w_a += w_a(i);
        //     sum_w_c += w_c(i);
        // }
        // gamma = std::sqrt(n_x / (1.0-w_a(0)) ); // adjusts the spread of sigma points     

        // Check if the sum of weights = 1
        // if( (std::fabs(sum_w_a - 1.0) > 0.00001) | (std::fabs(sum_w_c - 1.0) > 0.00001) ) {
        //     std::cerr << "sum_w_a: " << sum_w_a << std::endl;
        //     std::cerr << "sum_w_c: " << sum_w_c << std::endl;
        //     throw std::runtime_error("Sigma weights dont add up 1");
        // }

    }

    Eigen::MatrixXd gen_sigma(const Eigen::VectorXd& x_hat, const Eigen::MatrixXd& P) {
        Eigen::MatrixXd sigma_pts(n_x, 2 * n_x + 1); // columns are sigma points, rows are the states of each sigma
        sigma_pts.col(0) = x_hat;
        // Method 1:
        // Eigen::MatrixXd P_sqrt = P.sqrt();
        // Method 2:
        Eigen::MatrixXd P_sqrt = P.llt().matrixL();

        for(int i=1;i<=n_x;i++) {
            // sigma points in (+) direction of eigen vectors of P
            sigma_pts.col(i) = x_hat + gamma *  P_sqrt.col(i-1);
            // sigma points in (-) direction of eigen vectors of P
            sigma_pts.col(n_x+i) = x_hat - gamma *  P_sqrt.col(i-1);
        }

        return sigma_pts;
    }

    Eigen::MatrixXd motion_model_sigma(const Eigen::MatrixXd& sigma_pts, const Eigen::VectorXd& U) 
    {
        Eigen::MatrixXd sigma_trans(n_x,2*n_x+1); // sigma points gone through non-linear function

        // execute the motion model per each column = sigma point
        for(int i=0; i<sigma_pts.cols(); i++) {
            sigma_trans.col(i) = this->motion_model(sigma_pts.col(i), U, this->dt); 
        }

        return sigma_trans;
    }

    // This function is used for both sigma points of system states X and measurement variables Z
    // They possibly have different dimensions
    Eigen::MatrixXd observation_model_sigma(Eigen::MatrixXd& sigma_pts) 
    {
        // Note that transformed sigma points will have the dimension of the measured variables, hence R.cols
        Eigen::MatrixXd sigma_trans(n_z, 2*n_x+1); // sigma points gone through non-linear function

        // execute the observation model per each column = sigma point
        for(int i=0; i<sigma_pts.cols(); i++) {
            sigma_trans.col(i) = this->observation_model(sigma_pts.col(i)); 
        }

        return sigma_trans;
    }

    // This function is used for both state error covariance P and innovation covariance S
    // They possibly have different dimensions
    Eigen::MatrixXd calc_sigma_covar(const Eigen::VectorXd& x, const Eigen::MatrixXd& sigma_pts,
                                     const Eigen::VectorXd& w_c, const Eigen::MatrixXd& P_init) 
    {
        Eigen::MatrixXd P = P_init;
        for(int i=0; i<(2*n_x+1); i++) {
            Eigen::VectorXd diff = sigma_pts.col(i) - x; // (n,1)
            P = P + diff * diff.transpose() * w_c(i); // (n,n)
        }
        return P;
    }

    Eigen::MatrixXd calc_cross_cov(const Eigen::MatrixXd& sigma_x, const Eigen::VectorXd& x_hat_prio,
                                   const Eigen::MatrixXd& sigma_z, const Eigen::VectorXd& z_pred) {

        Eigen::MatrixXd C_xz = Eigen::MatrixXd::Zero(n_x,n_z); // C_xz = (n,m)
        
        for(int i=0; i<(2*n_x+1); i++) {
            Eigen::VectorXd diff_x = sigma_x.col(i) - x_hat_prio; // (n,1)
            Eigen::VectorXd diff_z = sigma_z.col(i) - z_pred; // (m,1)
            C_xz = C_xz + diff_x * diff_z.transpose() * w_c(i); // (n,m)
        }

        return C_xz;
    }

    void prediction_update(const Eigen::VectorXd& U) 
    {
        if(!initialized) {
            throw std::runtime_error("Filter is not initialized!");
        }

        // Generate sigma points around the last computed x_hat_post
        Eigen::MatrixXd sigma_pts = gen_sigma(x_hat_post, P_post);

        // Propagate the sigma points through non-linear process model
        Eigen::MatrixXd sigma_trans = motion_model_sigma(sigma_pts, U);

        // Produce predicted mean and covariance from propagated sigma points
        this->x_hat_prio = sigma_trans * w_a; // weighted average of sigma points = x_hat_prio
        P_prio = calc_sigma_covar(x_hat_prio, sigma_trans, w_c, Q); // weighted average of difference of sigma points and x_hat_prio
    }

    void innovation_update(const Eigen::VectorXd& Z) {
        
        // Generate sigma points around the x_hat_prio
        // ( a new set of sigma points since we iterated the process model)
        Eigen::MatrixXd sigma_pts = gen_sigma(x_hat_prio, P_prio);

        // Transform the sigma points through non-linear measurement model
        Eigen::MatrixXd sigma_trans = observation_model_sigma(sigma_pts);

        // Produce mean and covariance from the transformed sigma points
        Eigen::VectorXd z_predict = sigma_trans * w_a; // weighted average of sigma points = predicted measurement
        Eigen::MatrixXd S = calc_sigma_covar(z_predict, sigma_trans, w_c, R); // weighted average of difference of sigma points and z_predict

        // Calculate cross-covariance matrix
        Eigen::MatrixXd C_xz = calc_cross_cov(sigma_pts, x_hat_prio, sigma_trans, z_predict);

        // Calculate the Kalman gain
        Eigen::MatrixXd K = C_xz * S.inverse();

        // Update the mean and covariance estimates
        this->x_hat_post = this->x_hat_prio + K * (Z - z_predict);
        P_post = P_prio - K*S*K.transpose(); 

        // nis = calc_NIS(Z, z_predict, S);
    }

    // Return the current state
    Eigen::VectorXd get_state() {return x_hat_post;};

    // Return the current state error covariance
    Eigen::MatrixXd get_P() {return P_post;};        

    // is the filter initialized?
    bool initialized;

    // // Motion model (user defined)
    std::function<Eigen::VectorXd(const Eigen::VectorXd& x_hat_post, 
                                  const Eigen::VectorXd& u_measured, 
                                  const float& dt)> motion_model;

    // Observation model (user defined)
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> observation_model;

    // calculate Normalized Innovation Squared, based on predicted and actual measurement
    double calc_NIS(const Eigen::VectorXd& Z, const Eigen::VectorXd& Z_pred, const Eigen::MatrixXd& S) {
        return (Z - Z_pred).transpose() * S.inverse() * (Z - Z_pred); 
    }

    // get the calculated NIS value 
    double get_NIS() const{
        return nis;
    }

private:
    // Estimated states
    Eigen::VectorXd x_hat_prio, x_hat_post;

    float dt;

    int n_x, n_z; // number of state variables and measurement variables

    Eigen::MatrixXd P_post, P_prio, Q, R, I;

    float nis; // normalized innovation squared

    double alpha, beta, kappa, gamma; // control parameters for sigma spread
    Eigen::VectorXd w_a, w_c; // weights for the sigma points

};