1) using landmarks, where we exactly know where the landmark is on the map
2) using raycasted lidar detections in a maze-like map



- robot odometry to propagate the motion of each particle:
    - (previous particle state) + (noisy_ctrl_input * motion_model = dead_reckon) + exploration_noise
- assign weights to particles based on how closely the current range sensor measurements match with
the expected range sensor measurements of the particles.
    - expected range measurements of the particles are measured using the propagated particle state + map of the environment
- particle distribution is resampled based on the weights of the particles
    - increased likelihood of selection given to particles with higher weights

- At the end, determine the robot state estimate by taking average of all particle states.

* when the robot observes the landmarks in robot frame with the sensor:
    1) Find all landmarks within the particle's range, using the map info and particle's predicted state (obtained with motion model).
    2) use the particle's "predicted state" , to transform the robot-frame landmark observation to map-frame landmark observation.
    3) since the particle is in map frame, it already has a view of which landmarks can be observed from that state,
    and most importantly, #the unique id of the landmark#
    - associate the observed landmark (1) to predicted landmark (2) by nearest neighbor. (since both are in map frame now)
        - this will also assign a landmark id to the observed landmark. --> main purpose: giving map landmark id to observation
    - each observation will contribute to the weight of that particle:
    - for obs in observations:
        - weight_particle *= MLE(distance(particle_landmark_predict, robot_landmark_observed)). 
            a) multivariate gaussian dist
            b) w = w * gauss_likelihood(dz, math.sqrt(Q=0.2**2)) --> Q=uncertainty of landmark
    - right after weight update, resample the particles


* next stage: pole detection from the lidar pointcloud to derive the landmarks