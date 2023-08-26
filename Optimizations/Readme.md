# Iterative Methods In Optimization
In robotics, linear and non-linear least squares optimization problems are very common. Due to the size and complexity of these systems, many applications use iterative methods to find an approximate solution for such problems.

Matrices that we see commonly in these methods are:

**Gradient**: Vector field formed by partial derivatives of a *scalar function*. ($df/dx$, where $f$ is scalar and $x$ is vector). Also worded as 1st order derivate of a multivariate function.

When $f: R^n \rightarrow  R,$ 
gradient = $∇f: R^n \rightarrow R^n$

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Optimizations/resources/gradient.png" width=70% height=70%>

**Hessian**: Second order mixed partials of a scalar function. Also worded as 2nd order derivative of a multivariate function.

When $f: R^n \rightarrow  R,$ 
hessian = $∇f: R^n \rightarrow R^n$

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Optimizations/resources/hessian.png" width=70% height=70%>

**Jacobian**: Matrix formed by partial derivatives of a *vector function*. ($df/dx$, where $f$ and $x$ are vectors)

When $f: R^n \rightarrow  R^m,$ 
jacobian = $J_{i,j}: df_i/dx_j$

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Optimizations/resources/jacobian.png" width=70% height=70%>


Very commonly, we have our parameters $x: R^n$ that belongs to the mathematical model/system, and we make many ($m$) observations/measurements of this system.  

Hence in total, this actually builds a MIMO system of $R^n \rightarrow  R^m$. 

## Deep learning vs Robotics
In deep learning, *gradient descent* is a popular optimization method for training neural nets. Mostly, there is a *scalar* loss function obtained at the end. For each training sample, value of the loss function (forward) and the gradient of loss function w.r.t network weights (backward) is computed. The average of all the gradients per sample is then used to update the parameters.

#### Why not use Newton's method (or derivatives) in DL?
- Due to the number of training samples ($m$), its not feasible to calculate the Jacobian.
- Due to the size of network parameters ($n$), its not feasible to calculate the Jacobian & Hessian. 

-----
In robotics, we use the ***Jacobian*** to describe how our system with multiple observations are affected by change of our parameter set.

- Gradient descent: Uses the gradient of the **cost function $L=\sum(y_i - f_i(x))²$** to update model parameters.
    - gradient is computed per each sample and then averaged over number of samples, so $R^n$


- Gauss-Newton: Uses the jacobian of the **residual function $r_i = y_i - f_i(x)$**. Jacobian contains 1 sample per row, so $R^{m \times n}$.

- Levenberg-Marquardt: Smootly transitions between Gradient-Descent and Gauss-Newton
```Python
function levenberg_marquardt(f, J, x0, max_iterations, tolerance):
    '''
    f: function to minimize
    J: Jacobian matrix of f
    x0: initial guess
    max_iterations: max number of iterations
    tolerance: convergence tolerance
    '''
    x = x0
    lambda = small positive number (e.g., 0.001)

    for i = 1 to max_iterations:
        gradient = J(x).T * f(x)   # Transpose of Jacobian times the function value
        Hessian_approximation = J(x).T * J(x) + lambda * Identity_Matrix

        delta_x = -inverse(Hessian_approximation) * gradient

        if norm(delta_x) < tolerance:
            return x

        x_new = x + delta_x

        if cost(f, x_new) < cost(f, x):
            x = x_new
            lambda = lambda / factor   # Reduce lambda when getting closer
        else:
            lambda = lambda * factor   # Increase lambda when moving away

    return x
```

λ is referred as damping factor.

1. When λ is large, the LM behaves more like Gradient Descent sincethe term $λ \times I$ dominates, making the Hessian approximation more diagonal, and the algorithm moves mostly in the direction of steepest descent.
   
2. When λ is small, the LM behaves more like the Gauss-Newton since influence of $λ \times I$ becomes negligible compared to $J^TJ$, which is the Gauss-Newton Hessian approximation.
