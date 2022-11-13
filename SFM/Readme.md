# Bundle Adjustment
- Scale ambiguity is still there (if there's no known 3D landmark introduced somewhere in the chain)
- Most clearly stated in the abstract of this paper: https://rpg.ifi.uzh.ch/docs/ICCV09_scaramuzza.pdf:
    - *In structure-from-motion with a single camera, the scene can be only recovered up to a scale. In order to compute the absolute scale, one needs to know the baseline of the camera motion or the dimension of at least one element in the scene.*
- But the reconstruction is consistent within the chose world coordinates. However that construction would look exactly the same if everything in real 3D world had been upscaled by K amount.

 - Many of the structure from motion datasets provide 3D landmarks, initial camera poses and matched image features already.

 
Online
 1) Form visual odometry from successive frames
 2) Assign 3D pose to both camera and landmarks during V0
Offline
 3) Among all frames, find matching feature points
 4) run BA



 in slambook BA example:
 - observations: 2D pixel coordinates. Many observations. Each observation contains a camera_pose idx and 3D world point idx
 - vertices: (6D camera poses + camera intrinsics) + (3D landmark points)
 - edges: 2D feature measurement taken from a specific camera pose of a specific 3D landmark (connecting these 2 vertices)
 NOTE: 3D landmark poses are in world coordinates, therefore reprojectection error calculation needs to:
 - transform 3d world landmark to image plane :
    -world -> 3D-camera frame 
    - 3D-Camera frame -> image plane  (perspective proj)


<!-- Washington dataset: -->
camera_idx, unique_3d_pt_idx, current_cam_pix_x, current_cam_pix_y
...
repeated num_observation times

9 camera parameters
[R(3),t(3),f,k1,k2]
....
repeated for each camera index

unique_3d_world_points (in world coordinates)
[x,y,z coordinates] 
...
repeated for number of unique 3d points

## note that pixel coordinates of matched features do not exist in the data above.
## the 2D feature matches are actually represented as 3d point indices, 
## since going from 2D->3D is just a matter of perspective projection

4634530 + 9*1408 + 3*912229



## PIPELINE:
1) carla image saver: collect the depth and rgb images, and camera ground truth poses.
2) Pose Estimator: get camera poses with visual odometry using depth+rgb images, via pair-wise feature matching.
 --> Problem: When we look at rgb pointcloud vs keypoint pointcloud, since keypoint pointcloud is generated with floating point pixel coordinates,
 it does not align perfectly with the rgb cloud which is generated with integer pixel coordinates.