# Homography
Homography (*projective transformation*) in general is the transformation that takes you from one plane to another through projection. 

<img align="center" width="150" height="150" src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/homography_definition.png">


The main attractiveness of homography is due to the simplification it brings to certain transformations. When we have the assumption of 3D-points of interest being on the same plane `(Z=0)`, 3x4 camera matrix `(intrinsics * extrinsics)` simplifies into a (3x3) homography matrix.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/camera_matrix.png" width=30% height=30%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/homography_matrix.png" width=30% height=30%>

Note that there are 8 unique points in the homography matrix since the scale factor is arbitrary with the homogenous coordinates. `(h_33=1 or |H|=1)`

Homography matrix can be seen as a reduced form of the [essential matrix](../VisualOdometry/Indirect/matching/) which relates **any** set of points from one image to another (given a common 3d world point). 

In practice, it can be computed through feature point matching between 2 images and solving the over-deterministic system of equations through SVD. Once we the homography, we can compute where any point from one projective plane maps on to the second projective plane, ***without needing the 3D location or the camera parameters***

Most common occurances can be group into 3:
1. A planar surface captured by a camera (3d plane surface --> image plane).
2. The same planar surface viewed by 2 (or more) camera positions.
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/2_view_plane.png" width=30% height=30%>

3. A camera doing **pure rotation** around its projection axis, capturing arbirary world (not necessarily planar). 
    - This is approximately equal to capturing very distant objects with relatively smaller camera movement that is not necessarily pure rotation.
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/pure_rotation.png" width=30% height=30%>    

> **Note**
> As opposed to translational, rotational and scalar transformations (all of which combined called **affine transformations**), under homography transformation, parallel lines **are not** necessarily preserved.

## Usage Examples Of Homography
### Image Stitching (e.g. Panorama)
We have a camera undergoing pure rotation here and the aim is to merge the right image onto the left image.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/pure_rotation/rgb_00003.png" width=15% height=15%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/pure_rotation/rgb_00004.png" width=15% height=15%>

When the right image is transformed with homography, its borders are clipped by the original image dimensions. Hence, we need to:
1. Use homography to find the bounding corners of the warped right image.
2. Determine the required offset to bring the warped image bounding corners within > (0,0)
3. Use this offset to compute a new *shifted* homography matrix.
```c++
cv::Mat shift                   = (cv::Mat_<double>(3, 3) << 1.0, 0.0, extensionU, 0, 1, extensionV, 0, 0, 1);
cv::Mat shiftedHomographyMatrix = shift * homographyMatrix;
```

4. Apply the shifted homography matrix to the right image.
5. Merge the left and warped right image, where the size of the final image is determined by the union of the shifted+warped right image corners and the left image corners.
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/stitched_pure_rotation.png" width=30% height=30%>   

### Projecting 2D template image to planar surfaces

### Camera Pose Estimation
We have a **3D planar** object **whose dimensions are known**, and we're capturing its images from different positions. The goal is to estimate the pose of the camera at these locations, ***with respect to the reference frame of the 3D object***.

As mentioned above, homography contains a subset of the general camera matrix.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/homography_eq.png" width=17% height=17%>

After computing the homography, the above equation can be decomposed to extract  `R_1, R_2, t`.
1. Compute the keypoints of the reference planar object image.
2. Lift these 2D keypoints to 3D keypoints (with Z=0), since the dimensions of the 3D is known (meters/pixel).
3. Compute the homography between 3D reference object keypoints and the keypoints of the input images.
4. Using the camera, instrinsics we obtain
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/scale_ambiguity_1.png" width=17% height=17%>
Due to the scale ambiguity, for now we only have

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/scale_ambiguity_2.png" width=17% height=17%>

5. Normally `R3` can be obtained by the cross product of `R1xR2`.
    - However, due to the noise in our `R1` & `R2` estimations (`R1_hat & R2_hat`), we cannot directly find `R3` via cross product.
    - Instead, we first find the closest orthogonal basis vectors to `R1_hat` and `R2_hat`, via SVD.
    - Then get `R3` via `R1 x R2`
6. Since we know that 2-left columns of M are `R1` and `R2` which have **unit norm**, we use this info to recover `ʎ`.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/homography_matches.gif" width=40% height=40%>
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/bbox_homography.gif" width=40% height=40%>
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/homography_cam_pose.gif" width=40% height=40%>

## References
- https://www.uio.no/studier/emner/matnat/its/TEK5030/v20/forelesninger/lecture_6_1_pose-estimation.pdf