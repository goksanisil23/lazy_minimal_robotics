# Homography
Homography (*projective transformation*) in general is the transformation that takes you from one plane to another through projection. 

<img align="center" width="150" height="150" src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/homography_definition.png">


The main attractiveness of homography is due to the simplification it brings to certain transformations. When we have the assumption of 3D-points of interest being on the same plane `(Z=0)`, 3x4 camera matrix (= intrinsics*extrinsics) simplifies into a (3x3) homography matrix.
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/camera_matrix.png" width=30% height=30%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/homography_matrix.png" width=30% height=30%>
Note that there are 8 unique points in the homography matrix since the scale factor is arbitrary with the homogenous coordinates. `(h_33=1 or |H|=1)`

Homography matrix can be seen as a reduced form of the [essential matrix](../VisualOdometry/Indirect/matching/) which relates **any** set of points from one image to another (given common 3d world point). In practice, it can be computed through feature point matching between 2 images and solving the over-deterministic system of equations through SVD.

Most common occurances can be group into 3:
1. A planar surface captured by a camera (3d plane surface --> image plane).
2. The same planar surface viewed by 2 (or more) camera positions.
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/2_view_plane.png" width=30% height=30%>
3. A camera doing **pure rotation** around its projection axis, capturing arbirary world (not necessarily planar). 
    - This is approximately equal to capturing very distance objects with relatively smaller camera movement that is not necessarily pure rotation.
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/pure_rotation.png" width=30% height=30%>    

> **Note**
> As opposed to translational, rotational and scalar transformations (all of which combined called **affine transformations**), under homography transformation, parallel lines **are not** necessarily preserved.

## Usage Examples Of Homography
### Image Stitching (e.g. Panorama)
We have a camera undergoing pure rotation here and the aim is to merge the right image onto the left image. When the right image is transformed with homography, its borders are clipped by the original image dimensions. Hence, we need to:
1. Use homography to find the bounding corners of the warped right image.
2. Determine the required offset to bring the warped image bounding corners within > (0,0)
3. Use this offset to compute a new *shifted* homography matrix.
4. Apply the shifted homography matrix to the source image.
5. Merge the source and target image, where the size of the final image is determined by the union of the shifted+warped source image corners and the source image corners.

### Projecting 2D template image to planar surfaces

### Camera Pose Estimation
As mentioned above, homography contains a subset of the general camera matrix.
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/Homography/resources/homography_eq.png" width=30% height=30%>    
Once homography is computed with feature mathing and SVD, given the camera intrinsics, it can be decomposed to extract  `R_1, R_2, t`.