## Gauss-Newton
Gauss-Newton is a numerical method that is used to solve optimization problems which are formulated as least-squares. It is useful when we have some real-world sample points that are observed and want to fit a mathematical model onto those points, as well as possible. 

The main idea comes from minimizing the RMSE of the model function, with respect to the observation points, at the observation points, in an iterative way by utilizing Taylor's 1st order expansion.

Let's say that we have the ground truth function `y(x)` which produces the observed points, and we want to approximate that by `y_hat(x,p)`. Here `p` is the parameter of approximation (e.g. `px³`). The goal is to start with some initial guess for `p`, and iteratively add some h update value to p that brings `y_hat(x,p)` closer to `y` every step.

If we approximate `y_hat(x, p + h)` by 1st order Taylor expansion, at p:

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/GaussNewton/resources/taylor.png" width=70% height=70%>

and define the least-squares that we want to minimize at every observation point as

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/GaussNewton/resources/least_sq.png" width=40% height=40%>

In order to find an `h` that minimizes `S`, we use `dS/dh = 0` which locally minimizes S, at every update step. Using these 3 equations and some re-arrangement, we end up with

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/GaussNewton/resources/eq1.png" width=70% height=50%>

Therefore, to compute the value of h that moves `y_hat(x,p)` closer to `y(x)`, we solve for the above equation, substitude `p` with `p+h`, and repeat until we converge.

In practice, `y_hat` is dependent on several m parameters, so `p` and `h` have dimension `m x 1`, so that the equation above turns into

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/GaussNewton/resources/eq2.png" width=30% height=30%>

where J is the Jacobian matrix defined as the partial derivative of `y_hat` w.r.t each `j` element of `p` and `i` sample point x: `y` and `y_hat` contains values of `y(x)` and `y_hat(x,p)` evaluated at all sample points. Jacobian is constructed for the user's model at the observation points.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/GaussNewton/resources/jacobian.png" width=15% height=10%>

Note that this is a linear equation of variable `h`, which can be solved through various decomposition methods avaiable in Eigen library. It's called the **normal equation** or **Gauss-Newton equation**. Here, `J^T * J` can be interpreted as the *approximation of the second-order Hessian matrix*. The upside of Gauss-Newton method is that, unlike Newton's Method, it avoids the calculation of the Hessian matrix which might be complex.

## Example
Here we're trying to approximate the true function
`
y = exp(a*x*x + b*x + c)
`
under some Gaussian sample measurement noise.

The core of the implementation can be summarized by the following lines:

```c++
for (int iter = 0; iter < N_MAX_ITER; iter++) {
    for (int i_sample = 0; i_sample < N_SAMPLES; i_sample++) {
        // Get the sample prediction based on the current model
        y_est.at(i_sample) = evaluateModel(samples[i_sample].x);
        // Based on the current prediction and measured observation,
        // calculate the residual (error)
        residual(i_sample, 0) = (samples[i_sample].y - y_est.at(i_sample));
        // Calculate the Jacobian of the error at this sample point
        Eigen::VectorXd jacobian = calculateJacobian(samples[i_sample].x);
        // Update the aggregated Jacobian matrix for this sample
        for (int i_param = 0; i_param < N_PARAM; i_param++) {
            J(i_sample, i_param) = jacobian(i_param);
        }
    }
    // Solve the Gauss-Newton
    Eigen::MatrixXd JTJ = J.transpose() * J;
    // Remember that our linear equation to solve is
    // (J^T * J) * h = J^T * (y - y_hat)
    Eigen::VectorXd h =
        (JTJ).colPivHouseholderQr().solve(J.transpose() * residual);    
}
```


<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/GaussNewton/resources/gauss_newton.gif" width=70% height=70%>