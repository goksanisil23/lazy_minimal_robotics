# Unscented Kalman Filter
Unscented Kalman Filter uses a statistical linearization technique, that samples n points from the prior distribution of the non-linear system.

The main idea is to avoid making local analytical approximations like EKF, which makes a 1st order Taylor series linearization to make a forecast of the states, and
therefore might have a corrupt aposterior mean and covariance.

UKF avoids the derivatives by "deterministic sampling" where the state distribution is calculated by carefully choosing a minimal set of **sigma points**.

![](/UnscentedKalmanFilter/resources/sigma_points.png)

Special feature of sigma points is that, their weighted mean and variance agrees with the true mean and variance of the true Gaussian distribution. And when the states go through a non-linear transformation, weighted mean and variance of the sigma points (that goes through same non-linear transformation) can approximate the true mean and covariance to 3rd and 2nd orders respectively. This sampling method is called **Unscented Transform**.

![](/UnscentedKalmanFilter/resources/unscented_trans.png)

Sigma points are chosen in the direction of the eigen vectors of the state error covariance matrix P. The convention is to choose **(2n+1)** sigma points: 1 for the mean of the distribution (which is the last state estimate), and 2n for both directions associated to n eigen vectors.

## Process Update

Prediction update ( = state propagation = motion model), is executed by computing the weighted mean of the sigma points, which gives the apriori state estimate x_hat. The image below illustrates that, the weighted mean of sigmas give a more accurate estimation after the non-linear transformation, compared to directly passing the previous x_hat through the non-linear process model.

![](/UnscentedKalmanFilter/resources/state_propagation.png)

For computing the apriori state error covariance P, instead of using the Jacobians, the apriori state estimate found above is used.

## Measurement Update
Similar to process update, a new set of sigma points are calculated from x_hat_pre and P_pre, so that they can be passed through the non-linear measurement model.

![](/UnscentedKalmanFilter/resources/algo_ukf.png)

- S is the innovation covariance.
- C_sz is the innovation cross-covariance (between the estimated states and the measurements)
These 2 terms are used to calculate the Kalman gain.

A popular parametrization to choose the weights and sigma points:
![](/UnscentedKalmanFilter/resources/choosing_sigma.png)
where **Aj** is the j'th column of A, where P_k|k+1 = A*A^T. 

## Toy example:
Estimating the 2D position of a mobile robot from noisy GPS measurements and noisy control inputs
In the basic example shown below, the 2D pose of a robot modeled with constant turn rate and velocity magnitude (CTRV) is estimated via UKF. Robot reads the noisy 2D measurements (x-y position), 
as well as the noisy control inputs (v and w). System is non-linear due to motion model of the robot having trigonometric functions.

![](/UnscentedKalmanFilter/resources/ukf.gif)
