/**
 * @brief This is a simple example usage of the Extended Kalman Filter
 * It is used to estimate the 2D position(x,y) and heading(h) of a mobile robot, by using linear velocity(v) and angular rate(w) measurements.
 *
 * X (state vector) = [x y h v]' 
 * Z (measurement vector) = [x y]
 * U (input/control vector) = [v_in w_in]'
 * 
 * Motion Model: X_(k+1) = A_(k) * X_(k) + B_(k) * U_(k) = f(X,U) 
 * x_(k+1) = x_(k) + dt * cos(h) * v_in
 * y_(k+1) = x_(k) + dt * sin(h) * v_in
 * h_(k+1) = h_(k) + dt * w_in
 * v_(k+1) = v_in
 * 
 * Measurement Model: Z_(k) = H_(k) * X_(k) = h(X)
 * x_meas = x_(k)
 * y_meas = y_(k)
 * 
 * -------------------
 * Jacobian of Motion Model : F(x,u) = df/dX
 * --> Take partial derivatives of all equations in the motion model w.r.t. all states in X
 * dx/dx, dx/dy, dx/dh, dx/dv = [1, 0, -dt*sin(h)*v_in, dt*cos(h)] 
 * dy/dx, dy/dy, dy/dh, dy/dv = [0, 1, dt*cos(h)*v_in, dt*sin(h)]
 * dh/dx, dh/dy, dh/dh, dh/dv = [0 0 1 0]
 * dv/dx, dv/dy, dv/dh, dv/dv = [0 0 0 1]
 * 
 * Jacobian of Measurement (observation) Model: H(x) = dh/dX
 * dx_meas/dx, dx_meas/dy, dx_meas/dh, dx_meas/dv = [1, 0, 0, 0]
 * dy_meas/dx, dy_meas/dy, dy_meas/dh, dy_meas/dv = [0, 1, 0, 0]
 * 
 * 
 */

#include <iostream>
#include <vector>
#include <random>
#include <memory>

// #include "kalman.hpp"
#include <Eigen/Dense>

#include "matplotlibcpp.h"


Eigen::Vector4d motion_model(const Eigen::Vector4d& X, const Eigen::Vector2d& U, const float& dt) {
    Eigen::Matrix4d A;
    Eigen::Matrix<double, 4, 2> B;
    double h = X(2);

    A << 1.0 , 0.0 , 0.0 , 0.0
      , 0.0 , 1.0 , 0.0 , 0.0
      , 0.0 , 0.0 , 1.0 , 0.0
      , 0.0 , 0.0 , 0.0 , 0.0;

    B << dt*std::cos(h) , 0.0 
      , dt*std::sin(h) , 0.0 
      , 0.0 , dt 
      , 1.0 , 0.0;

    return (A * X + B * U);
}

Eigen::Vector2d observation_model(const Eigen::Vector4d& X) {
    Eigen::Matrix<double, 2, 4> C;
    
    C << 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0;

    return (C * X);
}

// Jacobian of the motion model
Eigen::Matrix4d jacobian_F(const Eigen::Vector4d& X_hat_prio, const Eigen::Vector2d& U, const float& dt) {
    Eigen::Matrix4d F;
    double v_in = U(0);
    double h = X_hat_prio(2);

    F << 1.0, 0.0, -dt * v_in * std::sin(h),  dt*std::cos(h),
         0.0, 1.0, dt * v_in * std::cos(h), dt*std::sin(h),
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0;

    return F;
}

// Jacobian of the measurement model
Eigen::Matrix<double, 2, 4> jacobian_H(const Eigen::Vector4d& X_hat_prio) {
    // normally jacobian is a function of current aprioristate estimate X, 
    // but it happened to be a constant matrix here.
    std::ignore = X_hat_prio; 
    
    Eigen::Matrix<double, 2, 4> H;

    H << 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0;

    return H;
}
/////////////
class EKF {

public: 

    EKF(const Eigen::MatrixXd& P0,
        const Eigen::MatrixXd& Q,
        const Eigen::MatrixXd& R,
        const float& dt ) : P_post(P0), P_prio(P0), Q(Q), R(R), dt(dt)
    {}

    void init(const Eigen::VectorXd& x0) {
        x_hat_prio = x0;
        x_hat_post = x0;

        I = Eigen::MatrixXd::Identity(x0.size(), x0.size());

        initialized = true;
    }


    void prediction_update(const Eigen::Vector2d& U) {
        if(!initialized) {
            throw std::runtime_error("Filter is not initialized!");
        }

        // Use the motion model to predict a-priori estimate
        this->x_hat_prio = this->motion_model(this->x_hat_post, U, this->dt);
        Eigen::MatrixXd jF = this->jacobian_F(this->x_hat_prio, U, this->dt);
        P_prio = jF * P_post * jF.transpose() + Q;
    }

    void innovation_update(const Eigen::VectorXd& Z) {
        Eigen::MatrixXd jH = this->jacobian_H(this->x_hat_prio);
        Eigen::VectorXd z_predict = this->observation_model(this->x_hat_prio);

        Eigen::MatrixXd K = P_prio * jH.transpose() * (jH * P_prio * jH.transpose() + R).inverse();
        this->x_hat_post = this->x_hat_prio + K * (Z - z_predict);
        P_post = (I - K * jH) * P_prio;
    }

    // Return the current state
    Eigen::VectorXd get_state() {return x_hat_post;};

    // Return the current state error covariance
    Eigen::MatrixXd get_P() {return P_post;};        

// private:
    // Estimated states
    Eigen::VectorXd x_hat_prio, x_hat_post;

    // is the filter initialized?
    bool initialized = false;

    // // Motion model (user defined)
    std::function<Eigen::Vector4d(const Eigen::Vector4d& x_hat_post, 
                                  const Eigen::Vector2d& u_measured, 
                                  const float& dt)> motion_model;

    // Observation model (user defined)
    std::function<Eigen::Vector2d(const Eigen::Vector4d&)> observation_model;

    // Jacobian of the motion model
    std::function<Eigen::Matrix4d(const Eigen::Vector4d& x_hat_prio,
                                  const Eigen::Vector2d& u_measured, 
                                  const float& dt)> jacobian_F;
    
    // Jacobian of the observation model
    std::function<Eigen::Matrix<double, 2, 4>(const Eigen::Vector4d& x_hat_prio)> jacobian_H;

    float dt;

    Eigen::MatrixXd P_post, P_prio, Q, R, I;


};

Eigen::Vector2d process_input_func() {
    double v = 1.0; // [m/s]
    double w = 0.1; // [rad/s]
    return Eigen::Vector2d(v,w);
}

///////////////////
int main() {


    float dt = 0.1; // simulation step time

    // Covariance matrices for the Kalman Filter
    Eigen::Vector4d q_variances(0.1, 0.1, 1.0/180.0*M_PI, 1.0); // variances of x,y,yaw,velocity in prediction
    Eigen::Matrix4d Q = q_variances.array().square().matrix().asDiagonal(); // Prediction update covariance
    Eigen::Vector2d r_variances(1.0, 1.0); // variances of x,y in observation
    Eigen::Matrix2d R = r_variances.array().square().matrix().asDiagonal(); // measurement update covariance
    Eigen::Matrix4d P0 = Eigen::MatrixXd::Identity(4, 4);

    EKF ekf_filter(P0, Q, R, dt);

    // Assign user defined models to the filter
    ekf_filter.motion_model = motion_model;
    ekf_filter.observation_model = observation_model;
    // Assign user defined Jacobians to the filter
    ekf_filter.jacobian_F = jacobian_F;
    ekf_filter.jacobian_H = jacobian_H;    

    // Some initial guess for the state we're trying to estimate
    Eigen::Vector4d x0(0, 0, 0, 0);
    ekf_filter.init(x0);

    // noise for the inputs v,w, and the measurement x,y
    std::default_random_engine rand_gen(0); // with a seed 0 for repeatibility
    std::normal_distribution<double> normal_dist_noise(0.0, 1.0);
    // scale the noise
    Eigen::Matrix2d meas_noise = Eigen::Vector2d(0.5, 0.5).array().square().matrix().asDiagonal();
    Eigen::Matrix2d input_noise = Eigen::Vector2d(1.0, 30.0/180.0*M_PI).array().square().matrix().asDiagonal();

    // True initial state variable
    Eigen::Vector4d X_true(0, 0, 0, 0);
    // Dead reckoning
    Eigen::Vector4d X_dead_reckon(0, 0, 0, 0);

    std::vector<double> x_hat_buffer{x0(0)}, y_hat_buffer{x0(1)};
    std::vector<double> x_true_buffer{X_true(0)}, y_true_buffer{X_true(1)};
    std::vector<double> x_dead_buffer{X_dead_reckon(0)}, y_dead_buffer{X_dead_reckon(1)};
    std::vector<double> z_meas_buffer_x, z_meas_buffer_y;
    std::vector<double> P_buffer_x, P_buffer_y;
    matplotlibcpp::figure_size(600,400);

    for(int i = 0; i < 500; i++) {

        // Calculate the true states of the system
        Eigen::Vector2d u = process_input_func();
        X_true = motion_model(X_true, u, dt);
        Eigen::Vector2d z = observation_model(X_true);

        // Add noise to the measurements
        Eigen::Vector2d z_meas = z + meas_noise *
                        Eigen::Vector2d(normal_dist_noise(rand_gen), normal_dist_noise(rand_gen)); // add noise to measurement
        Eigen::Vector2d u_meas = u + input_noise * 
                        Eigen::Vector2d(normal_dist_noise(rand_gen), normal_dist_noise(rand_gen)); // add noise to input

        // Execute EKF with the noisy measurements
        ekf_filter.prediction_update(u_meas);
        ekf_filter.innovation_update(z_meas);        

        // Calculate dead-reckoning --> just iterate the motion model with measurement (no correction)
        X_dead_reckon = motion_model(X_dead_reckon, u_meas, dt);

        // store the history
        x_hat_buffer.push_back(ekf_filter.get_state()(0));
        y_hat_buffer.push_back(ekf_filter.get_state()(1));
        x_true_buffer.push_back(X_true(0));
        y_true_buffer.push_back(X_true(1));      
        x_dead_buffer.push_back(X_dead_reckon(0));
        y_dead_buffer.push_back(X_dead_reckon(1));   
        z_meas_buffer_x.push_back(z_meas(0));
        z_meas_buffer_y.push_back(z_meas(1));
        P_buffer_x.push_back(ekf_filter.get_P()(0,0));
        P_buffer_y.push_back(ekf_filter.get_P()(1,1));

        matplotlibcpp::clf();
        matplotlibcpp::named_plot("x hat", x_hat_buffer, y_hat_buffer);
        matplotlibcpp::named_plot("x true", x_true_buffer, y_true_buffer, "y");
        matplotlibcpp::named_plot("x dead", x_dead_buffer, y_dead_buffer, "k");
        matplotlibcpp::named_plot("z meas", z_meas_buffer_x, z_meas_buffer_y, ".g");
        matplotlibcpp::grid(true);
        matplotlibcpp::legend();
        matplotlibcpp::ylabel("State variable unit");
        matplotlibcpp::xlabel("iterations");
        matplotlibcpp::pause(0.001);
    }

        // Rendering
        matplotlibcpp::clf();
        matplotlibcpp::named_plot("Px", P_buffer_x);
        matplotlibcpp::named_plot("Py", P_buffer_y);
        matplotlibcpp::grid(true);
        matplotlibcpp::legend();
        matplotlibcpp::ylabel("P");
        matplotlibcpp::xlabel("iterations");    
        matplotlibcpp::pause(0);       


    return 0;
}