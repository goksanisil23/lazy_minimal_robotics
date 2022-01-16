# Extended Kalman Filter
To deal with non-linear systems, EKF uses a 1st order Taylor approximation to **locally** linearize a non-linear system.
Here, we choose the linearization point as the most recent state estimate, our known input, and zero noise. 
Linearization is established by computing the Jacobian matrices, which contain the 1st order partial derivatives of the non-linear system equations.

For implementation, we refer to: https://www.cse.sc.edu/~terejanu/files/tutorialEKF.pdf

Here, there are 2 Jacobians calculated: 
- Jacobian of the motion model w.r.t. the states
- Jacobian of the measurement model w.r.t the states.

*(Since we considered the noise in motion model and measurement model as "additive" white noise, Jacobians
w.r.t process and measurement noises reduce to identity matrices, so they don't appear in covariance equations. Additive process white noise is not the best assumption, but simlifies the Jacobian)*

![](/ExtendedKalmanFilter/resources/ekf_overview.png)


In this implementation, Jacobians are calculated by hand. However, an interesting paradigm called "automatic differentiation", which can algorithmically
compute the derivative of basically any function could be utilized in the future as well.

A practical rule of thumb on choosing process noise covariance values: Choose half of the maximum value expected from the state variable, as the process noise. (e.g. if we're estimating the acceleration of a vehicle, which is ~max 6 m/s² in normal traffic, choose sigma (std.dev.) as 3 m/s²)

## Consistency Check
One way to check for the validity of the hand-picked noise parameters, is through consistency check.
At each timestep, we calculate the **predicted measurement mean (z_k+1|k) and covariance (S_k+1|k)**. Then we receive the actual measurement **(z_k+1)** for that timestep.

![](/ExtendedKalmanFilter/resources/consistency_pic.png)

A filter is consistent if it provides a realistic information about uncertainty. Under-estimation and over-estimation of the magnitude of error covariances can be both problematic. 

Here, we can utilize <ins>Normalized Innovation Squared (NIS)</ins>, as a consistency measure:

![](/ExtendedKalmanFilter/resources/NIS.png)

**I**nnovation: Difference between predicted measurement vs. the actual measurement.

**N**ormalized: Put in relation to innovation covariance matrix S.

NIS gives a simple scalar value, that follows Chi-Squared distribution. A low value for chi-square means there is a high correlation between your 2 sets of data. In Kalman filter, these 2 sets of data are the predicted measurement and the actual measurement. A chi-squared table could be used to check against the NIS value.

![](/ExtendedKalmanFilter/resources/chi_squared_table.png)

**df** is the degrees of freedom and corresponds to the size of the measurement space. Values in the table says that, for dof 3, "*statistically in 95% of all cases, your NIS will be higher than 0.352, and in 5% of all cases NIS will be higher than 7.815*".   

## Toy example: Estimating the 2D position of a mobile robot from noisy GPS measurements and noisy control inputs
In the basic example shown below, the 2D pose of a robot modeled with constant turn rate and velocity magnitude (CTRV) is estimated via EKF. Robot reads the noisy 2D measurements (x-y position), 
as well as the noisy control inputs (v and w). System is non-linear due to motion model of the robot having trigonometric functions.

![](/ExtendedKalmanFilter/resources/ekf.gif)