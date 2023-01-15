# Optical Flow

The goal of optical flow is to generate a 2D flow field that describes how the pixels in the image are moving in time to create a representation of the dynamics of the scene.

Optical flow has 2 main assumptions:
- Displacement of pixels (`dx` & `dy`) and time step (`dt`) are small 
- The brightness of an entity in pixel space in 2 successive frames remains constant over time. (reasonable given 1st assumption)

These assumptions allow simplification of higher order terms in Taylor series:

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OpticalFlow/resources/taylor_1.png" width=50% height=50%>

Since we assume the brightness remains constant over small dt for the point of interest, we can further simplify the notation as:

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OpticalFlow/resources/optical_flow_constraint.png" width=50% height=50%>

where `u = dx/dt, v=dy/dt` is the optical flow and `Ix, Iy, It` are the changes in the intensity in x,y directions and in time, that can be computed with finite differences:

`Ix = 1/4 * {I(x+1,y,t) + I(x+1,y+1,t) + I(x+1,y,t+1) - I(x+1,y+1,t+1)} - 1/4 * {I(x,y,t) + I(x,y+1,t) + I(x,y,t+1) - I(x,y+1,t+1)}`

Optical flow by nature is an underconstrained problem, since 1 constraint equation above has 2 unknowns: `u & v`

To overcome this, Lukas-Kanade approach assumes all pixels *within a small neighborhood* (3x3 window) have the same motion field, which gives a number of equations for unknowns u & v, that can be solved by least-squares.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OpticalFlow/resources/lucas_kanade_matrix.png" width=30% height=50%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OpticalFlow/resources/lukas_kanade_solution.png" width=30% height=50%>

Note that for optical flow to work, `A^T*A` must be well-conditioned:
- Both eigen values are not too small: Meaning no change in intensities of pixels over time: e.g. textureless sky
- One eigenvalue is not too dominant over the other: Corresponds to edge-like structures. Cannot know if edge is moving diagonally or perpendicularly.

**What happens when motion is large?**
In the case of large motion, the main optical flow constraint is not satisfied anymore. To overcome this, a **resolution pyramid** is used to start with a coarse estimation and propagate to finer estimations. The idea is that, a large motion of a scene point in high res image, will be smaller if the same scene is represented in a lower res image (since there arent as many pixels). Therefore, we start with estimating the optical flow in the smallest resolution image pairs (image[t], image[t+1]), and we use the optical flow result of the previous step to propagate the optical flow to higher resolution images.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OpticalFlow/resources/klt_pyramid.png" width=70% height=50%>

## Sparse Optical Flow
In sparse optical flow, we only track a limited number of feature points. So given a keypoint `I_(k)(x,y)` in image `[k]`, we want to find where this keypoint has moved in image `[k+1]`: `I_(k+1)(x+Δx,y+Δy)`. This is especially useful in visual odometry applications where visual landmarks are needed to be tracked reliably across the frames for triangulation. 

Here, instead of looking for a 1-to-1 pixel intensity matching for this keypoint, we look at a certain image patch (-4,+4 pixels) around the keypoint.

In this implementation, we use the aforementioned image pyramids and Gauss-Newton optimization in the following way:

>- Calculate the keypoints in image `[k]`
>- Build the image pyramids for frame `[k]` & `[k+1]` 
>- Scale the keypoints of image `[k]` for the top level of the pyramid (lowest resolution)
>- For each layer of the pyramid:
>    - For each keypoint:
>        - Get initial `Δx, Δy` from the previous pyramid layer.
>        - For N Gauss-Newton iterations (or until convergence):
>            - For each pixel inside the keypoint patch window:
>                - Calculate the residual: `I_(k)(x,y) - I_(k+1)(x+Δx,y+Δy)`
>                - Accumulate the bias, cost and hessian based on the residual
>            - Update Δx, Δy based on the solution of `update = bias * inv(Hessian)`


<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/OpticalFlow/resources/optical_flow_sparse.png" width=50% height=50%>
