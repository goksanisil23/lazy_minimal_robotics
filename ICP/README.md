# Iterative Closest Point (ICP)
ICP is a method to solve the rigid-transform (rotation + translation) problem between 2 sets of points. These 2 sets can be both partially or fully overlapping.
The "iterative" aspect of ICP is due to the fact that point-wise correspondences between 2 sets of data are (mostly) unknown. 

## SVD Based Point-to-Point ICP
If the correspondences were known, finding the optimal rotation and translation between those 2 sets can be solved by using [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm) **in 1 step**. 2 important elements here are:
- Covariance matrix of 2 sets: Defined by the inner product of 2 sets of **origo-centered points**: (3xN) * (Nx3) = (3x3)
- Singular Value Decomposition: U*V^T of SVD is intuitively the direction of the rotational difference between 2 sets of data.
```c++
Eigen::Matrix3d H = centered_input * centered_target.transpose(); // H = 3x3
Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
Eigen::Matrix3d R = svd.matrixV() * I * svd.matrixU().transpose();
```

But since we don't know the correspondences, ICP makes certain assumptions. One of the simplest flavors of ICP takes closest points between 2 sets as correspondences. For certain number of iterations, this assumption is most probably wrong. (In the example below, this can be seen by 1-to-many correspondences). However, if 2 sets were reasonably close, this approach gets us closer to the actual correspondences, and therefore to the actual transformation. 

- Correlation matrix H is computed between the estimated correspondences between the input (which iteratively transforms) and target point sets
- ICP stops when the correspondence from the previous iteration doesn't change anymore.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ICP/3D/resources/2d_ICP.gif" width=50% height=50%>

A considerable speed improvement can be gained by storing the target point set in a K-D tree and making a nearest neighbor search on the tree to find the correspondences. Here, we used [MRPT library](https://github.com/vioshyvo/mrpt) for KNN search.

## Least-squares Point-to-Plane ICP
In order to solve the rigid-body alignment more efficiently, we can consider the fact that, mostly, points measured in real world belong to some surface. Therefore, instead of minimizing point-to-point distances, we can instead minimize point-to-surface distance. Practically, this requires projecting the point-to-point error vector on to the associated surface normal (which is simply taking the dot product `{x_source-x_target)·x_target_normal}`. Note that, here we still need to find the point-wise correspondences between 2 sets, in order to compute point (set 1) to normal (set 2) projection.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ICP/3D/resources/point_to_plane.png" width=40% height=40%><img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ICP/3D/resources/tangent.png" width=35.5% height=35%> 

Point-to-normal minimization breaks the SVD- based closed-form solution though. So instead, a least-squares approach can be used to minimize the cost function, shown below:

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ICP/3D/resources/equation_point_to_plane.png" width=50% height=50%>

Since the rotation matrix in M contains nonlinear trigonometric functions, linear least-squares techniques cannot be directly applied. As a workaround, via small-angle approximation sin(ѳ) ≅ ѳ, cos(ѳ) ≅ 1, R can be linearized, which gives an approximate transformation matrix as

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ICP/3D/resources/m_approx.png" width=40% height=40%>

If all N pairs of correspondes are considered, solving (1) becomes equivalent to solving
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ICP/3D/resources/linear.png" width=30% height=50%>
<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ICP/3D/resources/x.png" width=28% height=50%>

which is a standard linear least-squares problem that can be solved by SVD, where SVD is used to compute the pseudo-inverse of A.


The algorithm looks as follows:
- Compute the normals of the target cloud
- Find the closest correspondences btw. input and target
- Construct the linearized least squares Ax=B, according to 8 in [[4]](https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf )
- Solve the linear system via SVD
- Apply the SVD result to the input.
- Check if the correspondences have converged. If not, repeat from 2nd step.

<img src="https://raw.githubusercontent.com/goksanisil23/lazy_minimal_robotics/main/ICP/3D/resources/3d_point_to_plane.png" width=50% height=50%>

### References
- [1] https://en.wikipedia.org/wiki/Kabsch_algorithm
- [2] https://nbviewer.org/github/niosus/notebooks/blob/master/icp.ipynb#ICP-based-on-SVD
- [3] https://gfx.cs.princeton.edu/proj/iccv05_course/iccv05_icp_gr.pdf
- [4] https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf 