## Kalman Filter
- P = state error covariance: Encrypts the error covariance that the filter thinks the estimated error has.
- Process noise covariance (Q) and measurement noise covaraince (R) can change each time step or measurement in reality, but we assume constant here. 

Kalman filter basically estimates a process as a feedback control: The filter estimates the process at some time, and then obtains a feedback as noisy measurements.

Time update equations (PREDICTOR): Projects the current state and error covariancea estimates, forward in time, to obtain  a-priori estimate for the next time step.

Measurement update (CORRECTOR): Incorporates new measurement into a-priori estimate to obtain an a-posteriori estimate.

This folder contains the implementation as shown below, from https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

![](/KalmanFilters/resources/Kalman_predict_correct.png)

A toy example where a constant state variable is being estimated with noisy measurements is provided. Below, the affect of varying measurement noise covariance can be seen. With a higher R, system becomes slower to respond but more prone to the noise, whereas with smaller R, system is very quick to "trust" the noisy measurements. 

![](/KalmanFilters/resources/kalman_R_comparison.png)