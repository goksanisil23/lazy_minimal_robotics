# Particle Filter
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ParticleFilter/resources/particle_filter_main.gif" width=100% height=100%>

[Kalman filter](../KalmanFilter) and [it's variants](../ExtendedKalmanFilter) work well when the motion (process) model is **not highly non-linear** and the **process & measurement noises can be approximated nearly Gaussian**. However for arbitrary distributions, a non-parametric representation where probability distribution is obtained through samples (particles) instead can be better suited.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ParticleFilter/resources/prob_dist_approx.png" width=50% height=50%>

To represent the probability distribution of the states, a set of weighted samples each of which represent a state hypothesis is used.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ParticleFilter/resources/prob_dist_eq.png" width=50% height=50%>
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ParticleFilter/resources/monte_carlo_approx.png" width=50% height=50%>

## Importance sampling for Resampling
- Goal: Generating samples from *target distribution (f)* (pdf of our robot's state)
- Challenge: Target distribution is unknown, so we cannot generate samples from it.
- Workaround: Use an alternative *proposal distribution (g)* (e.g our motion model + some Gaussian noise), generate samples, do a correction based on the observed difference between *g* and *f*.

Intuitively, weight assignment can be seen as **survival of the fittest**: *weight = f/g*

> **Note**
> Particle filter, just like Kalman filter, is still a Bayesian filter, which means the state estimation is performed by combining a prior state probability (iteration with motion model) with a statistical model for a measurement (likelihood).

**Prediction:** motion model and control commands + exploration noise. 
*This noise is to account for the wrongness of the motion model and control input, and basically randomly exploring the vicinity to correct for such errors.*
**Correction:** sensor measurements. Given that my robot would be where this particle is, how likely is it to obtain particle's observation, given the robot's observation.
Based on this likeliness, the particle receives an importance weight.

> **Warning**
> To represent larger uncertainties more particles are needed. Similarly, higher dimension state estimates require more samples.

## Implementation for 2D localization
This example is using the known map of the environment, noisy odometry info and noisy landmark measurements for localization. 

- Robot odometry to propagate the motion of each particle:
    - (previous particle state) + (noisy_ctrl_input * motion_model = dead_reckon) + (exploration_noise)
- Assign weights to particles based on how closely the current range sensor measurements match with the expected range sensor measurements of the particles.
    - Expected range measurements of the particles are measured using the propagated particle state + map of the environment
- Particle distribution is resampled based on the weights of the particles
    - Increased likelihood of selection given to particles with higher weights
- At the end, determine the robot state estimate by taking average of all particle states.

The "exploration" phase of the particles and the resampling through robot observations become more clear when filter prediction step is 10x compared to sensor update step:
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ParticleFilter/resources/particle_filter_convergence.gif" width=100% height=100%>

- When the robot observes the landmarks in robot frame with the sensor:
    1. Find all landmarks within the particle's range, using the map info and particle's predicted state (obtained with motion model).
    2. use the particle's "predicted state" , to transform the robot-frame landmark observation to map-frame landmark observation.
    3. since the particle is in map frame, it already has a view of which landmarks can be observed from that state, and most importantly, the unique id of the landmark
    - associate the observed landmark (1) to predicted landmark (2) by nearest neighbor. (since both are in map frame now)
        - this will also assign a landmark id to the observed landmark. --> main purpose: giving map landmark id to observation
    - each observation will contribute to the weight of that particle:
    - for obs in observations:
        - weight_particle *= MLE(distance(particle_landmark_predict, robot_landmark_observed)).
    - right after weight update, resample the particles

## Shortcomings
It is not guaranteed that the particle filter will converge to the global solution, nor recover from global localization failures. After some duration, particles converge around a single pose and unable to recover if this happens to be incorrect. As particle filter is a stochastic algorithm, it may accidentally discard the correct particles during the resampling step. This is most likely when number of particles are small or insufficient diversification under relatively low process variance (exploration factor) or when the map is relatively large.

In the recovery example below, we're just resetting the filter to it's initial conditions where particles are randomly distributed within the map.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ParticleFilter/resources/particle_filter_reset.gif" width=100% height=100%>

In uniform looking environments, robot's observations will be very similarly associated to particle predictions in different parts of the map, which can lead the filter to converge to a local-minima.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ParticleFilter/resources/particle_filter_uniform_map.gif" width=100% height=100%>


1) using landmarks, where we exactly know where the landmark is on the map
2) using raycasted lidar detections in a maze-like map

* next stage: pole detection from the lidar pointcloud to derive the landmarks
* using raycasted lidar detections in a maze-like map --> instead of doing 1-to-1 matching, do pattern matching of landmarks

## References
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7826670/
- https://online.stat.psu.edu/stat505/book/export/html/636 