# Ceres
Ceres is an optimization library for solving problems that can be formulated as non-linear least-squares. It provides tools to construct an optimization problem one term at a time and a solver API that controls minimization algorithm.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/Ceres/resources/residual_block.png" width=50% height=50%>

In Ceres lingo, `f_i` is a cost function (residual block) and `p_i` is a loss function, which can be used to reduce the influence of outliers. `[x_i_1, ..., x_i_k]` is called a parameter block, which is a group of scalars that we're trying to find the best value for.

The way a user utilizes Ceres is through defining the parameter block, defining the calculation of the residual block and choosing how to compute Jacobians and the solver method. A powerful method called automatic differentiation saves the user from calculating the Jacobians of the model manually. Automatic differentiation libraries usually implement it via operator overloading, where primitive arithmetic operations are augmented to handle the derivative computation for each term.

The way Ceres enables this is through
- Having the cost function defined in a class/struct
- overloading the function call operator
- templating the function call operator

so that internally `<double>()` is called when residual is needed, and `<Jet>()` is called when the Jacobian is needed.

For each observation (sample) point, the user creates a new instance of the residual class for that point, and then adds it to the optimization problem. This is equivalent to constructing the concatanated Jacobian matrix in the Gauss Newton example.

## Example
In the example below, we are trying to estimate the parameters of a ground truth function`y = exp(mx+c)` which is observed under some Gaussian noise.
In one of the solvers, we're utilizing the Cauchy loss function in order to reduce the effect of outlier samples.

The core of the implementation can be summarized by the following lines:
```c++
// Define the residual within the function call () operator as Ceres wants
// in a templated manner for internal automatic differentiation for error Jacobian calculation
struct ExponentialResidual {
  ExponentialResidual(double x, double y) : x_(x), y_(y) {}

  template <typename T>
  bool operator()(const T *const m, const T *c, T *residual) const {
    residual[0] = y_ - evaluteModelAtSamplePoint(m[0], c[0], x_);
    return true;
  }

private:
  const double x_;
  const double y_;
};

/* ... */

// Construct the Ceres optimization problem
// by adding residual blocks constructed with observation samples
ceres::Problem ceres_problem;
for (int i = 0; i < N_SAMPLES; i++) {
ceres_problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
        new ExponentialResidual(x_gt.at(i), y_obs.at(i))),
    nullptr, &m, &c);
}

// Define solver options and solve
ceres::Solver::Options solver_options;
solver_options.max_num_iterations = 25;
solver_options.linear_solver_type = ceres::DENSE_QR;

ceres::Solve(solver_options, &ceres_problem, &solver_summary);

```


<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/Ceres/resources/ceres.png" width=50% height=50%>