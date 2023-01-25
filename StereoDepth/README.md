## Stereo Camera Setup for Depth Estimation
A fundamental limitation of mono-camera vision systems is the lack of scale/depth information. Even in the multi-view monocular visual odometry case where multiple images from a single camera are used for tracking/matching common image features over time for solving PnP or the fundamental matrix, without having the knowledge of some 3D landmark information that is up-to-scale, the correct depth/scale of the 3D scene cannot be recovered.

The stereo-camera vision systems tackles this problem. The missing degrees of freedom (depth), comes from the known baseline between the cameras.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/StereoDepth/resources/stereo_overview.png" width=30% height=50%>

With the perspective projection equations above, we get 4 equations for the 3 unknown X,Y,Z real-world coordinates of the common landmark point seen by both cameras. Solving for X,Y,Z gives

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/StereoDepth/resources/x_y_z_from_stereo.png" width=30% height=50%>

- Disparity is inversely proportional to *depth Z*.
- Disparity is proportional to the *baseline*.

* Note that in the above equations, we see the disparity in the **u** axis of the image, which is due to this stereo setup being horizontally offset.
* From the Z equation, it can be seen that for measuring further distance objects, it's beneficial to have a large baseline. Otherwise, the disparity `(u_l - u_r)` will be very small which
    - practically might be smaller than the pixel size that cannot be detected
    - is more prone to imperfections in feature matching between left and right cameras

The benefit of having a horizontally (or vertically) and axis aligned baseline is that, in disparity computation, it reduces the area that we search for feature matches between the camera pairs to a horizontal strip.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/StereoDepth/resources/find_match_disparity.png" width=30% height=50%>

This is due to both `v_l` and `v_r` having the same perspective projection:  `v_l = v_r = f_y * Y / Z + o_y`
In order to avoid the pixel on the left camera being wrongly matched to a pixel on the right camera's horizontal strip, a block comparison is applied instead.

> **Note**
> This can be seen as a **special case of the epipolar geometry**. In general, when 2 cameras that are separated by an *arbitrary translation and rotation* see the same 3D landmark, the search space for matching this landmark's pixel from one camera to the other reduces to a single line called **epipolar line**. When the cameras have a relative orientation, epipolar line will also be rotated with respect to the image axes. 

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/StereoDepth/resources/epipolar_line.png" width=30% height=50%>

Searching for feature correspondes between axis-aligned cameras is much simpler since it does not require computation of the epipolar line and finding pixels along that line, as the epipolar line becomes horizontally (or vertically) level with the camera's x/y-axes. 

In practice though, not all stereo-camera setups can be perfectly axis aligned on the baseline. To still reap the benefits of axis-aligned stereo setup, **stereo rectification** can be applied to make the epipolar lines horizontal. This process re-projects both camera image planes onto a common plane parallel to the baseline between the cameras. It uses the [Fundamental Matrix](../VisualOdometry/Indirect/matching/) to do so.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/StereoDepth/resources/rectification_diagram.png" width=30% height=50%>

In summary, there are 2 main problems to solve in stereo-depth estimation:
1. Finding baseline, and camera intrinsics.
2. Finding matching pixel coordinates `(u_l,v_l) & (u_r,v_r)` for the landmark. -> for computing disparity

## Results
For testing, a simulated camera from Carla and the stereo dataset from Middlebury is used.

#### Original left & right pair
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/StereoDepth/resources/middlebury_motorcycle_left.png" width=30% height=50%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/StereoDepth/resources/middlebury_motorcycle_right.png" width=30% height=50%>

#### Disparity with: Block matching, Semi-global block matching, Weighted Least Squares Filter applied to SGBM, Final Depth Map 
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/StereoDepth/resources/motorcycle_result.png" width=60% height=70%>

#### Confidence Map (white = high confidence in disparity)
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/StereoDepth/resources/confidence_map.png" width=50% height=70%>

#### 3D-reconstructed pointcloud from the depth map
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/StereoDepth/resources/motorcycle_stereo_3d.png" width=60% height=70%>