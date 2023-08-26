"""
This script implements simple versions of 
1) Gradient descent
2) Gauss-Newton
3) Levenberg-Marquardt 
optimization techniques for solving non-linear least squares problems.

Given noisy observations of the true system y = e^(a*x^2+ b*x + c), we try to find estimated values for
the parameters a, b, c. 

Run: python opt_comparison.py
"""

import matplotlib.pyplot as plt
import numpy as np

# True parameters
a_true = 0.3
b_true = 0.5
c_true = 0.4

a_init = 0.0
b_init = 0.0
c_init = 0.0

MAX_ITERS = 100
TOL = 1e-6

# Generate synthetic data
np.random.seed(0)
x = np.linspace(-2, 2, 100)
w = np.random.normal(0, 0.1, size=x.shape)  # noise


def model(x, a, b, c):
    return np.exp(a * x**2 + b * x + c)


y = model(x, a_true, b_true, c_true) + w


def loss(y, y_pred):
    return np.sum((y - y_pred) ** 2)


######################### GRADIENT-DESCENT #########################
# This is the gradient of the sum-of-squared-errors cost function for gradient descent
# SSE = sum(y_i - f_i(x))^2
class GradientDescent:
    def __init__(self):
        self.a_est, self.b_est, self.c_est = a_init, b_init, c_init
        self.learning_rate = 0.006  # AT THIS VALUE WE OSCILLATE AROUND THE SOLUTION!
        self.converged = False

    def gradient(self, x, residuals, a, b, c):
        da = -2 * residuals * (x**2) * (np.exp(a * x**2 + b * x + c))  # dCost/da
        db = -2 * residuals * (x) * (np.exp(a * x**2 + b * x + c))  # dCost/db
        dc = -2 * residuals * (np.exp(a * x**2 + b * x + c))  # dCost/dc

        n = len(residuals)
        return np.sum(da) / n, np.sum(db) / n, np.sum(dc) / n

    def step(self):
        residuals = y - model(x, self.a_est, self.b_est, self.c_est)
        da, db, dc = self.gradient(x, residuals, self.a_est, self.b_est, self.c_est)
        self.a_est -= self.learning_rate * da
        self.b_est -= self.learning_rate * db
        self.c_est -= self.learning_rate * dc

        y_pred_curr = model(x, self.a_est, self.b_est, self.c_est)
        # print(f"loss: {loss(y,y_pred_curr)}")

        if np.linalg.norm(np.array([da, db, dc])) < TOL:
            self.converged = True


######################### GAUSS-NEWTON #########################
class GaussNewton:
    def __init__(self):
        self.a_est, self.b_est, self.c_est = a_init, b_init, c_init
        self.converged = False

    @classmethod
    def jacobian(self, x, residuals, a, b, c):
        J = np.zeros((len(x), 3))
        J[:, 0] = -((x**2) * (np.exp(a * x**2 + b * x + c)))
        J[:, 1] = -((x) * (np.exp(a * x**2 + b * x + c)))
        J[:, 2] = -((np.exp(a * x**2 + b * x + c)))
        return J

    def step(self):
        residuals = y - model(x, self.a_est, self.b_est, self.c_est)
        J = self.jacobian(x, residuals, self.a_est, self.b_est, self.c_est)
        update = np.linalg.inv(J.T @ J) @ J.T @ residuals
        self.a_est -= update[0]
        self.b_est -= update[1]
        self.c_est -= update[2]
        y_pred_curr = model(x, self.a_est, self.b_est, self.c_est)
        # print(f"loss: {loss(y,y_pred_curr)}")

        if np.linalg.norm(update) < TOL:
            self.converged = True


# print(f"Gauss-Newton: a={a_est}, b={b_est}, c={c_est}")


######################### LEVENBERG-MARQUARDT ########################
class LevenbergMarquardt:
    def __init__(self):
        self.params = [a_init, b_init, c_init]
        self.lambd = 10
        self.factor = 10
        self.I = np.eye(3)
        self.delta = None
        self.converged = False

    def step(self):
        residuals = y - model(x, *self.params)
        J = GaussNewton.jacobian(x, residuals, *self.params)
        grad = np.dot(J.T, residuals)
        hessian_approx = np.dot(J.T, J) + self.lambd * self.I
        self.delta = np.linalg.solve(hessian_approx, -grad)

        params_new = self.params + self.delta

        residuals = y - model(x, *self.params)
        residuals_new = y - model(x, *params_new)

        ## A) Simple lambda update
        # Compare the costs
        if np.linalg.norm(residuals_new) < np.linalg.norm(residuals):
            self.params = params_new
            self.lambd = self.lambd / self.factor
        else:
            self.lambd = self.lambd * self.factor

        if np.linalg.norm(self.delta) < TOL:
            self.converged = True


gd = GradientDescent()
gn = GaussNewton()
lm = LevenbergMarquardt()

for i in range(MAX_ITERS):
    if not gd.converged:
        gd.step()
    if gd.converged:
        print(f"GD converged in {i} steps")
        break
print(f"Gradient Descent: a={gd.a_est}, b={gd.b_est}, c={gd.c_est}")

for i in range(MAX_ITERS):
    if not gn.converged:
        gn.step()
    if gn.converged:
        print(f"GN converged in {i} steps")
        break
print(f"Gauss NEwton: a={gn.a_est}, b={gn.b_est}, c={gn.c_est}")

for i in range(MAX_ITERS):
    if not lm.converged:
        lm.step()

    if lm.converged:
        print(f"LM converged in {i} steps")
        break
print(f"Levenberg Marquardt: a={lm.params[0]}, b={lm.params[1]}, c={lm.params[2]}")
