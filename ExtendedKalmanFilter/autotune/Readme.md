# Auto-tuninig of EKF noise 
Many times in practice, performance of the Kalman filters are determined by the tuning accuracy of the process noise and measurement noise parameters. If we have the possibility to access ground truth either for the measurement of the internal states, it can be used for tuning those parameters.

Here, we have used the Ceres library to form the least-squares problem, where the tuning parameters are Q and R matrix diagonal entries.
Optimization is implemented as a single residual block with `N*num_state` dimensions (where `N` is the number of time samples in the dataset) since the current state of the filter is determined by the previous state which are dependent on the templated covariance parameters  we're optimizing.

### Other tuning references:
http://alexthompson.ai/tuning-EKF-process-noise/