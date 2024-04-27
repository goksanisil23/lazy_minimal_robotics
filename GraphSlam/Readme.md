# GraphSlam

Simple graph-based non-linear least squares optimization framework to combine different sensor measurements to jointly optimize for the pose of the robot as well as the landmarks (map) in the environment.

### Measurements
We have:
- Noisy IMU measurements (linear & angular velocity with some random distribution)
- Noisy landmark measurements (range & bearing with some random distribution and ID)

For simplicity, we discard the landmark association problem, meaning each landmark measurement comes with a unique identifier.

At each timestep, parameters (states) we're solving for:
- All observed landmarks up until that point
    - This is feasible for this environment with limited size.
- Most recent 20 robot poses. (updated via sliding window)
    - Since we're solving at each step at 60Hz, we need to limit the number of parameters while keeping enough for consistency of the solution.

### Constraints
*(Also called factors, cost functions)*

Using the 2 measurements mentioned above, we introduce 2 **observation constraints** to the graph. Each of these constraints formulize a residual using the parameters we're going to be solving for in the graph together with the measurement values that are treated as constants during optimization.

- Odometry Constraint
    - Odometry measurements provide `delta_pose` between 2 timestamps, in robot frame.
    - We keep a history of robot poses as states in the graph (P<sub>k</sub>,P<sub>k-1</sub>,...,P<sub>k-19</sub>)
    - Each `delta_pose` allows us to create a ***prediction error (residual)*** by comparing P<sub>k</sub> to (P<sub>k-1</sub> + `delta_pose`)

- Landmark Constraint
    - For each unique robot pose, there's a set of landmark measurements.
    - Both global landmark poses and global robot poses are states in the graph.
    - For a given landmark measurement in robot frame, using the states in the graph, we can construct an ***expected landmark observation***, for that unique landmark that's being measured.
    - Comparing expected landmark observation to the landmark measurement allows to create a prediction error (residual).


<img align="center" src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/GraphSlam/resources/graph_slam.png" width=50% height=50%>


Ceres allows to describe and solve this graph optimization without explicitly deriving the Jacobians. Limited size of the optimization parameters (20 robot & ~150 landmark poses) allows us to run the optimization at each tick.

We prune the older robot poses and associated landmark observations from the graph in a sliding window, at each step.

For comparison, we visualize the solution of the graph-slam together with dead-reckoning which simply accumulates the delta-motion from the IMU-odometry.


<img align="center" src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/GraphSlam/resources/graph_slam.gif" width=50% height=50%>