1) using landmarks, where we exactly know where the landmark is on the map
2) using raycasted lidar detections in a maze-like map



- robot odometry to propagate the motion of each particle:
    - (previous particle state) + (odometry) + noise
- assign weights to particles based on how closely the current range sensor measurements match with
the expected range sensor measurements of the particles.
    - expected range measurements of the particles are measured using the propagated particle state + map of the environment
- particle distribution is resampled based on the weights of the particles
    - increased likelihood of selection given to particles with higher weights

- At the end, determine the robot state estimate by taking average of all particle states.


* for sensor sim: instead of running raycasting, check if the landmark is within the radius of the particle/vehicle.
    * transform the detections in the particle coordinate system to map coordinate system
    * 


* when the robot observes the landmarks in robot frame with the sensor:
    1) use that particle's "predicted state" (obtained with motion model), to transform the robot-frame landmark to map-frame landmark.
    2) since the particle is in map frame, it already has a view of which landmarks can be observed from that state,
    and most importantly, #the unique id of the landmark#
    - associate the observed landmark (1) to predicted landmark (2) by nearest neighbor.
        - this will also assign a landmark id to the observed landmark.
    - assign the weight for each particle based on the distance between the associated landmark distances of predicted and observed cases.


* next stage: pole detection from the lidar pointcloud to derive the landmarks