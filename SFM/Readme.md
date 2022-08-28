# Bundle Adjustment
- Scale ambiguity is still there (if there's no known 3D landmark introduced somewhere in the chain)
- Most clearly stated in the abstract of this paper: https://rpg.ifi.uzh.ch/docs/ICCV09_scaramuzza.pdf:
    - *In structure-from-motion with a single camera, the scene can be only recovered up to a scale. In order to compute the absolute scale, one needs to know the baseline of the camera motion or the dimension of at least one element in the scene.*
- But the reconstruction is consistent within the chose world coordinates. However that construction would look exactly the same if everything in real 3D world had been upscaled by X.

 - Many of the structure from motion datasets provide 3D landmarks, initial camera poses and matched image features already.

 
Online
 1) Form visual odometry from successive frames
 2) Assign 3D pose to both camera and landmarks during V0
Offline
 3) Among all frames, find matching feature points
 4) run BA



 in slambook BA example:
 - observations: 2D pixel coordinates. Many observations. Each observation contains a camera_pose idx and 3D world point idx
 - vertices: (3D camera poses+camera intrinsics), (3D landmark poses)
 - edges: 2D feature measurement taken from a specific camera pose of a specific 3D landmark (connecting these 2 vertices)
 NOTE: 3D landmark poses are in world coordinates
 therefore reprojectection error calculation needs to:
 - transform 3d world landmark to image plane :
    -world -> 3D-camera frame 
    - 3D-Camera frame -> image plane  (perspective proj)


<!-- Washington dataset: -->
camera_idx, unique_3d_pt_idx, current_cam_pix_x, current_cam_pix_y
...
num_observation times repeated

9 camera parameters repeated for each camera index
[R(3),t(3),f,k1,k2]

unique_3d_world_points, repeated for number of unique 3d points