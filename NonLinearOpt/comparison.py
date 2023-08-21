import matplotlib.pyplot as plt
import numpy as np

# True parameters
a_true = 0.3
b_true = 0.5
c_true = 0.4

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


def gradient(x, residuals, a, b, c):
    da = -2 * residuals * (x**2) * (np.exp(a * x**2 + b * x + c))
    db = -2 * residuals * (x) * (np.exp(a * x**2 + b * x + c))
    dc = -2 * residuals * (np.exp(a * x**2 + b * x + c))

    n = len(residuals)
    return np.sum(da) / n, np.sum(db) / n, np.sum(dc) / n


# Gradient Descent
lr = 0.006  # AT THIS VALUE WE OSCILLATE AROUND THE SOLUTION!
a_est, b_est, c_est = 0.0, 0.0, 0.0
for i in range(MAX_ITERS):
    residuals = y - model(x, a_est, b_est, c_est)
    da, db, dc = gradient(x, residuals, a_est, b_est, c_est)
    a_est -= lr * da
    b_est -= lr * db
    c_est -= lr * dc

    y_pred_curr = model(x, a_est, b_est, c_est)
    print(f"loss: {loss(y,y_pred_curr)}")
    if np.linalg.norm(np.array([da, db, dc])) < TOL:
        print(f"GD converged in {i} steps")
        break

print(f"Gradient Descent: a={a_est}, b={b_est}, c={c_est}")


# # #########################3
def jacobian(x, residuals, a, b, c):
    J = np.zeros((len(x), 3))
    J[:, 0] = -((x**2) * (np.exp(a * x**2 + b * x + c)))
    J[:, 1] = -((x) * (np.exp(a * x**2 + b * x + c)))
    J[:, 2] = -((np.exp(a * x**2 + b * x + c)))
    return J


# # Gauss-Newton
a_est, b_est, c_est = 0, 0, 0
for _ in range(MAX_ITERS):  # usually Gauss-Newton converges faster than GD
    residuals = y - model(x, a_est, b_est, c_est)
    J = jacobian(x, residuals, a_est, b_est, c_est)
    update = np.linalg.inv(J.T @ J) @ J.T @ residuals
    a_est -= update[0]
    b_est -= update[1]
    c_est -= update[2]
    y_pred_curr = model(x, a_est, b_est, c_est)
    print(f"loss: {loss(y,y_pred_curr)}")

    if np.linalg.norm(update) < TOL:
        print(f"GN converged in {i} steps")
        break

print(f"Gauss-Newton: a={a_est}, b={b_est}, c={c_est}")


# # ##############################################


# def hessian(params, x, y_observed):
#     a, b, c = params
#     y_pred = model(x, a, b, c)

#     daa = np.sum(y_pred * x**4)
#     dab = np.sum(y_pred * x**3)
#     dac = np.sum(y_pred * x**2)
#     dbb = np.sum(y_pred * x**2)
#     dbc = np.sum(y_pred * x)
#     dcc = np.sum(y_pred)

#     return np.array([[daa, dab, dac], [dab, dbb, dbc], [dac, dbc, dcc]])


# def newton_method(x, y_observed, max_iters=10, tol=1e-6):
#     params = np.array([0.0, 0.0, 0.0])  # starting parameters
#     for i in range(max_iters):
#         grad = gradient(x, y_observed, params[0], params[1], params[2])
#         hess = hessian(params, x, y_observed)

#         delta_params = -np.linalg.inv(hess).dot(grad)
#         params += delta_params

#         if np.linalg.norm(delta_params) < tol:
#             break

#     return params


# a_fit, b_fit, c_fit = newton_method(x, y)
# y_fit = model(x, a_fit, b_fit, c_fit)

# print(f"Newton's Method: a={a_est}, b={b_est}, c={c_est}")
