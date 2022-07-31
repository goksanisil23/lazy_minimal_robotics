# Visual Odometry Via Sift/Orb Feature Matching
In monocular visual odometry, in order to estimate the 3D pose of the camera, all methods require an observed 3D landmark at some point along the pipeline. However, extraction of the depth information is not possible from a single frame. But still, with the RGBD or stereo-camera setups, it's possible to obtain the depth information per pixel. Given that the depth info for at least 1 of the frames is somehow available, we will look into 2 setups:
- **2D-2D:** Recovering 3D camera pose only up-to-scale
- **3D-2D:** Given a 3D landmark in previous frame, estimate 3D pose with scale.

**Camera Extrinsics (R & t)**: Project from world coordinates (object point in 3d space) --> Camera coordinates (originated at the location of camera) 

**Camera Intrinsics (K)**: Project from camera coordinates --> Image coordinates

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/world_camera_image.png" width=30% height=50%>  <img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/world_to_pixel_eq.png" width=27% height=50%>

## Pose Estimation via Fundamental Matrix (2D-2D)
- Epipolar geometry: When 2 camera views are seeing same 3D world point (this can be either from a stereo camera setup, or a single moving camera seeing the same point at 2 instances in time), triangulation of the camera centers with the 3D world point leads to certain geometric relations that are studied under epipolar geometry.

- Epipolar line: When we form a ray from camera 0 origin to scene point, all possible points on this line projected onto camera 1 creates an epipolar line.

- Epipole: Points where the baseline intersects the two image planes.

- Epipolar plane: Plane defined by the 3D world point seen by both cameras, and the centers of 2 cameras.

- Epipolar constraint: Dot product of the vector normal to epipolar plane and the vector of world point P in one of the camera coordinate frames, equals to 0.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/epipolar_constraint.png" width=30% height=50%>

Epipolar constraint, together with the inter-camera relation **`x_l = R*x_r + t`** forms the equation for the essential matrix. Fundamental property of essential matrix is that it can be decomposed into the translation and rotation components.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/essential_matrix_eq.png" width=30% height=50%>

However, xl and xr above are the 3D coordinates of the same scene point in 2 camera frames, which we do not have, hence we cannot directly use essential matrix to recover pose. 

Instead, we have the corresponding image coordinates. Incorporating perspective pinhole projection equations (3d->2d) into the essential matrix equation gives: 

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/fundamental_matrix_1.png" width=30% height=50%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/fundamental_matrix_2.png" width=30% height=50%>

where K_l and K_r are the intrinsics of the cameras. Note that fundamental matrix F and kF describe the same epipolar geometry. So, F is only defined up-to-scale. With this loss of dof, we need at least 8 pixel correspondences between cameras to solve for F.

In this implementation, we used the ground truth from CARLA to get the true scale of the relative translation t between successive camera frames.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/viso_essential_matrix.gif" width=100% height=50%>

## Pose Estimation via Perspective-N-Point (3D-2D)
In order to recover the camera pose from known 3D-to-2D correspondences, PnP can be used. OpenCV's solvePnPRansac implementation solves PnP as follows:
1) Find an initial guess of R & t via Direct Linear Transform (DLT). 

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/DLT_1.png" width=40% height=50%>

The extrinsics matrix T=(R|t) above contains 12 unknowns, and each feature point provides 2 linear constraints. Therefore, linear solution of the matrix T can be found by at least 6 pairs of matching feature points. If there are more than 6 pairs, SVD can also be used to find closed-form least-square solution of the overdetermined equation.

***So are we done?***   
DLT approach assumes that camera pose has 12 degrees of freedom (unknowns), whereas in reality it has only 6 due to rotation having 3dof instead of 9dof. Therefore, DLT solution does not consider R being in 3D rotation group SO(3). Although QR decomposition can be used to extract a proper rotation matrix R, since the closed-form solution itself does not contain this constraint, DLT solution is not final.

2) Use DLT as an initial guess for non-linear least-squares optimization:

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/least_square_PNP.png" width=30% height=50%>

where K is the intrinsics, P is the 3D-world point, s is the scale factor and u is the pixel coordinates in homogenous form.
Minimization of the residual given above corresponds to minimizing the reprojection error shown below. Since we know the matching feature points in both images, we're trying to find the best translation and rotation parameters that minimizes the distance between re-projected feature point p_2_hat and the actual matched feature point p_2.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/reprojection_error.png" width=30% height=50%>

3) The reprojection error described above is considered for all "n" matching feature pairs. With RANSAC, we introduce a threshold so that points having higher reprojection error are considered as outliers and eliminated from error minimization. 

------

In camera calibration, PnP is utilized to find the camera extrinsics with respect to an object with known dimensions (like checkerboard) in 3D. Note that in the calibration case, the extrinsics of the camera will be with respect to the 3D-world origin from which the 3D points of the known object is measured (e.g. top left corner of the checker board being (0,0,0) )

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/world_to_camera.png" width=30% height=50%>   <img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/PNP_odom.png" width=30% height=10%>


In the case of visual odometry however, we're interested in how the camera has moved w.r.t previous pose of the camera. So instead, we place the 3D-world origin at the previous frame's camera center. For PnP, this means:
- Using 3D coordinates of the objects obtained from image ***[k-1]***, which are w.r.t [k-1] camera origin. 
- Finding the corresponding 2D pixel coordinates in ***[k]***, for the *same 3D object points* represented in [k-1] camera origin.

In practice, this requires 2 main steps (in addition to solving PnP)
1) Getting the 3D coordinates of feature points from image [k-1]
2) Matching the features between frames [k-1] & [k]


In this implementation, ORB feature detection and description is chosen to associate keypoints among the frames. To minimize the false matches of features, Lowe's ratio method is used in which, we ensure that there is considerable distance difference between the best match and 2nd best match. If the top 2 candidates are too similar in distance score, that match is not used.

To get the 3D coordinates of the detected keypoints, inverse camera projection is used. However, as seen from the image [2], this only provides world coordinates up to scale since a ray originating from the camera center and passing through a ray, has infinitely many 3D-world candidates along that ray. In order to recover the scale, we utilize the measurements from the depth camera.

```C
depth = prev_depth_img.at(prev_keypoint.y, prev_keypoint.x);
x_world = (prev_keypoint.x - c_x) * depth / f_x;
y_world = (prev_keypoint.y - c_y) * depth / f_y;
world_pt = (x_world, y_world, depth); 
```

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/Indirect/matching/resources/viso_pnp.gif" width=100% height=50%>

## Pitfalls
- Although both approaches use distance based filtering to reduce false matches, this does not entirely remove all especially for the textures in the image that are quite similar.


### References

- https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf 
- https://github.com/gaoxiang12/slambook-en/blob/master/slambook-en.pdf
- https://www.youtube.com/watch?v=_rfKoEBGK7E&list=PL2zRqk16wsdoCCLpou-dGo7QQNks1Ppzo&index=9 