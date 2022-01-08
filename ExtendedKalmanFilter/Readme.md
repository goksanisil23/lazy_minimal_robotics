## Extended Kalman Filter
To deal with non-linear systems, EKF uses a 1st order Taylor approximation to **locally** linearize a non-linear system.
The linearization point, is the most recent state estimate. 
Linearization is established by computing the Jacobian matrices, which contain the 1st order partial derivatives of the non-linear system equations.

For implementation, refer to: https://www.cse.sc.edu/~terejanu/files/tutorialEKF.pdf 

In this implementation, Jacobians are calculated by hand. However, an interesting paradigm called "automatic differentiation", which can algorithmically
compute the derivative of basically any function could be utilized in the future as well.

![](/ExtendedKalmanFilter/resources/ekf_overview.png)

## Toy example: Estimating the 2D position of a mobile robot from noisy GPS measurements and noisy control inputs
In the basic example shown below, the 2D pose of a robot making a circle is estimated via EKF. Robot reads the noisy 2D measurement inputs, 
as well as the noisy control inputs. System is non-linear due to motion model of the robot having trigonometric functions.

![](/ExtendedKalmanFilter/resources/ekf.gif)