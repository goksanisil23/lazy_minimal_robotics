# Dense Visual Odometry With Optical Flow

The goal of optical flow is to generate a 2D flow field that describes how the pixels in the image are moving in time to create a representation of the dynamics of the scene.

Optical flow has 2 main assumptions:
- Displacement of pixels (`dx` & `dy`) and time step (`dt`) are small 
- The brightness of the image point in 2 successive frames remains constant over time. (reasonable given 1st assumption)

These assumptions allow simplification of higher order terms in Taylor series:

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/dense/resources/taylor_1.png" width=30% height=50%>

Since we assume the brightness remains constant over small dt for the point of interest, we can further simplify the notation as:

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/dense/resources/optical_flow_constraint.png" width=30% height=50%>

where `Ix, Iy, It` are the changes in the intensity in x,y directions and in time, that can be computed with finite differences:

`Ix = 1/4 * {I(x+1,y,t) + I(x+1,y+1,t) + I(x+1,y,t+1) - I(x+1,y+1,t+1)} - 1/4 * {I(x,y,t) + I(x,y+1,t) + I(x,y,t+1) - I(x,y+1,t+1)}`

Optical flow by nature is an underconstrained problem, since 1 constraint equation above has 2 unknowns: `u & v`

To overcome this, Lukas-Kanade approach assumes all pixels *within a small neighborhood* (window) have the same motion field, which gives a number of equations for unknowns u & v, that can be solved by least-squares.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/dense/resources/lucas_kanade_matrix.png" width=30% height=50%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/VisualOdometry/dense/resources/lucas_kanade_solution.png" width=30% height=50%>

Note that for optical flow to work, `A^T*A` must be well-conditioned:
- Both eigen values are not too small: Meaning no change in intensities of pixels over time: e.g. textureless sky
- One eigenvalue is not too dominant over the other: Corresponds to edge-like structures. Cannot know if edge is moving diagonally or perpendicularly.

**What happens when motion is large?**
In the case of large motion, the main optical flow constraint is not satisfied anymore. To overcome this, a resolution pyramid is used to start with a coarse estimation and propagate to finer estimations. The idea is that, a large motion of a scene point in high res image, will be smaller if the same scene is represented in a lower res image (since there arent as many pixels). Therefore, we start with estimating the optical flow in smallest resolution image pairs (image[t], image[t+1]), and we use the OF result of the previous step to propagate the optical flow to higher resolution images.