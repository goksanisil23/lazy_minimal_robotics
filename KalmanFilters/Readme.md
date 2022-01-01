## Kalman Filter
- P = state error covariance: Encrypts the error covariance that the filter thinks the estimated error has.
- Process noise covariance (Q) and measurement noise covaraince (R) can change each time step or measurement in reality, but we assume constant here. 

Kalman filter basically estimates a process as a feedback control: The filter estimates the process at some time, and then obtains a feedback as noisy measurements.

Time update equations (PREDICTOR): Projects the current state and error covariancea estimates, forward in time, to obtain  a-priori estimate for the next time step.

Measurement update (CORRECTOR): Incorporates new measurement into a-priori estimate to obtain an a-posteriori estimate.

![](KalmanFilters/KalmanFilter/resources/Kalman_predict_correct.png)