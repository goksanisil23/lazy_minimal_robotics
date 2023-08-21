# Iterative Methods In Optimization
In robotics, linear and non-linear least squares optimization problems are very common. Due to the size and complexity of these systems, many applications use iterative methods to find an approximate solution for such problems.


**Gradient**: Vector field formed by partial derivatives of a *scalar function*. ($df/dx$, where $f$ is scalar and $x$ is vector). Also worded as 1st order derivate of a multivariate function.

When $f: R^n \rightarrow  R,$ 
gradient = $∇f: R^n \rightarrow R^n$

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/resources/gradient.png" width=70% height=70%>

**Hessian**: Second order mixed partials of a scalar function. Also worded as 2nd order derivative of a multivariate function.

When $f: R^n \rightarrow  R,$ 
hessian = $∇f: R^n \rightarrow R^n$

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/resources/hessian.png" width=70% height=70%>

**Jacobian**: Matrix formed by partial derivatives of a *vector function*. ($df/dx$, where $f$ and $x$ are vectors)

When $f: R^n \rightarrow  R^m,$ 
jacobian = $J_{i,j}: df_i/dx_j$

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/NonLinearOpt/resources/jacobian.png" width=70% height=70%>


Very commonly, we have our parameters $x: R^n$ associated to our **scalar** model/system $R^n \rightarrow  R$, and we make many ($m$) observations/measurements of this system, making the overall output $R^m$. 

Hence in total, this actually builds a MIMO system of $R^n \rightarrow  R^m$. 

In deep learning, *gradient descent* is a popular optimization method for training neural nets. Mostly, there is a *scalar* loss function obtained at the end. For each training sample, value of the loss function (forward) and the gradient of loss function w.r.t network weights (backward) is computed. The average of all the gradients per sample is then used to update the parameters.

Due to the number of training samples, its not feasible to calculate the Jacobian.
Due to the size of network parameters, its not feasible to calculate the Hessian. 


In robotics, therefore we use the ***Jacobian*** to describe how our system with multiple observations are affected by change of our parameter set.

- Gradient descent: Uses the gradient of the **cost function** to update model parameters.
    - gradient is computed per each sample and then averaged

- Gauss-newton: Uses the jacobian of the **residual function**. Jacobian contains each sample per row, so the sample size affects Jacobian dimensions.