# Structure from Motion using Bundle Adjustment
When we have multiple images of a given scene from different view points, the 3d structure of the scene can be recovered. In the Visual Odometry section, we have seen that the pose of a moving camera can be estimated, by matching the keypoints from one frame to the next. In bundle adjustment, this concept is taken one step further: For a given 3d world point (landmark), if we can determine the corresponding keypoint of the landmark in multiple camera images (ideally 2+), we can jointly optimize for the position of the landmarks and the camera poses by using the graph structure of the problem.

## Prerequisites for Bundle Adjustment
- Initial estimates for the camera poses (6-dof) in world frame
- Position of the landmarks (3-dof) in world frame
- Observations that relate the cameras to the landmarks

The core idea of bundle adjustment is to minimize the reprojection error given multiple measurements of the same landmark from multiple viewpoints. Here, measurements are the pixel coordinates of a landmark in the image frame, with a camera index. Therefore, a crucial pre-step of bundle adjustment is to associate keypoints in multiple images to each other. Once a 3d landmark coordinate can be associated to pixel coordinates in multiple images, a graphical bundle adjustment problem can be formulated in variuous optimization libraries (g2o, Ceres, etc.)

### Note about scale in Structure From Motion
- Scale ambiguity is still there (if there's no known 3D landmark introduced somewhere in the chain)
- Most clearly stated in the abstract of this paper: https://rpg.ifi.uzh.ch/docs/ICCV09_scaramuzza.pdf:
    - *In structure-from-motion with a single camera, the scene can be only recovered up to a scale. In order to compute the absolute scale, one needs to know the baseline of the camera motion or the dimension of at least one element in the scene.*
- But the reconstruction is consistent within the chosen world coordinates. However that construction would look exactly the same if everything in real 3D world had been upscaled by K amount.
- Many of the structure from motion datasets provide 3D landmarks, initial camera poses and matched image features already.

## Implementation
 - Edges (observations): 2D pixel coordinate + camera index (which contains this keypoint) + 3D landmark index (which forms this keypoint in this image)
 - Vertices: 6D camera poses, 3D landmark points, camera intrinsics (optional)

 Therefore an edge is connecting a camera index to a landmark index, for each measurement (pixel keypoint)
 
 **NOTE**: 3D landmark points are in world coordinates, therefore reprojectection error calculation needs to:
 - transform 3d world landmark to image plane:
    - world -> 3D-camera frame 
    - 3D-Camera frame -> image plane (perspective proj)

### Bundle adjustment data format
We follow the "Bundle ADjustment In Large" Washington dataset format. (https://grail.cs.washington.edu/projects/bal/)

### Pipeline:

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/SFM/resources/bundle_adjustment_diagram.png" width=50% height=50%>

1) carla image saver: collect the depth and rgb images, and camera ground truth poses.
2) **Pose Estimator**: Estimate camera poses with visual odometry using depth + rgb images, via pair-wise feature matching.
3) **Feature Matcher**: Find matching keypoints *in at least 3 images*, which will constitute a landmark. Generate the 3D world position of the landmark by using position of one of the cameras seeing that landmark. This step outputs a "bundle adjustment dataset" txt file containing:
- observations (pixel keypoint, camera index, landmark index)
- 6D camera poses in world frame
- 3D landmark points in world frame
 which will be used in next step.
4) **Bundle Adjuster**: Jointly optimize for the camera poses and landmark positions by minimizing the landmark reprojection error.

### Result


<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/SFM/resources/rgb_cloud_viso.png" width=50% height=50%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/SFM/resources/rgb_cloud_g2o.png" width=50% height=50%>


<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/SFM/resources/rgb_cloud_viso_2.png" width=50% height=50%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/SFM/resources/rgb_cloud_g2o_2.png" width=50% height=50%>