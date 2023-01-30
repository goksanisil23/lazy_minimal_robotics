# Lazy Minimal Robotics
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ParticleFilter/resources/particle_filter_convergence.gif" width=30% height=30%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ICP/3D/resources/3d_point_to_plane.gif" width=18% height=20%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/SFM/resources/sfm_before_after.png" width=52% height=30%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/homography_cam_pose.gif" width=25% height=25%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OccupancyGrid/resources/ogrid_lidar.gif" width=25% height=24%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OpticalFlow/resources/sparse_oflow_traffic.gif" width=25% height=25%>

Minimal implementations for personally interesting ideas &amp; algorithms that can be useful for robotics applications, in C/C++.

### Menu

- [x] [Gauss-Newton curve fitting](/NonLinearOpt/GaussNewton)
- [x] [Ceres curve fitting](/NonLinearOpt/Ceres)
- [x] [g2o Unary Edge: Graph curve fitting](/NonLinearOpt/GraphOpt#unary-edge-example)
- [x] [g2o Binary Edge: 1D robot localization](/NonLinearOpt/GraphOpt#binary-edge-example)
- [x] [Kalman Filter](/KalmanFilter)
- [x] [Extended Kalman Filter](/ExtendedKalmanFilter)
- [x] [Auto-tuning EKF with Ground Truth](/ExtendedKalmanFilter/autotune)
- [x] [Unscented Kalman Filter](/UnscentedKalmanFilter)
- [x] [Particle filter](/ParticleFilter)
- [x] [ICP](/ICP)
- [x] [Homography](/Homography)
- [x] [Optical Flow (Sparse)](/OpticalFlow)
- [x] [Indirect Monocular Visual Odometry: Feature Matching](/VisualOdometry/Indirect/matching)
- [x] [Indirect Monocular Visual Odometry: Feature Tracking](/VisualOdometry/Indirect/tracking)
- [x] [Stereo Vision Depth Estimation](/StereoDepth)
- [ ] Direct Monocular Visual Odometry
- [ ] Direct with GPU
- [x] [Structure from Motion with Bundle Adjustment](/SFM)
- [ ] Structure from Motion with Pose Graph Optimization
- [ ] Dense Reconstruction via Multi-View Stereo
- [ ] Factor graph object tracking
- [x] [2D Occupancy Grid From 3D lidar](/OccupancyGrid)
- [ ] 3D Voxel Occupancy Grid (static and dynamic)
- [ ] 3D signed distance field (euclidean and truncated)
- [ ] Single & Multi object tracking
- [ ] Optimal Assignment (hungarian algo)
- [ ] Ant colony optimization