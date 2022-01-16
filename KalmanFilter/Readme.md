## Kalman Filter
Kalman Filter estimates the states of a system (or process) that can be described by linear system equations. It's a resursive least-squares estimation
that allows to combine information from motion (process) model with sensor measurements.

It can be used to estimate both the unmeasurable or the measurable but noisy states of the system. 

- P = state error covariance: Encrypts the error covariance that the filter thinks the estimated error has.
- Process noise covariance (Q) and measurement noise covaraince (R) can change each time step or measurement in reality, but we assume constant here. 

Kalman filter basically estimates a process as a feedback control: The filter estimates the process at some time, and then obtains a feedback as noisy measurements.

Time update equations (PREDICTOR): Projects the current state and error covariance estimates, forward in time, to obtain a-priori estimate for the next time step.

Measurement update (CORRECTOR): Incorporates new measurement into a-priori estimate to obtain an a-posteriori estimate.

This folder contains the implementation as shown below, from https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

![](/KalmanFilter/resources/Kalman_predict_correct.png)

# Toy example: Estimating a static value from noisy measurement readings
A simple example where a constant state variable is being estimated with noisy measurements is provided. Below, the affect of varying measurement noise covariance can be seen. With a higher R, system becomes slower to respond but more prone to the noise, whereas with smaller R, system is very quick to "trust" the noisy measurements. 

![](/KalmanFilter/resources/kalman_R_comparison.png)

