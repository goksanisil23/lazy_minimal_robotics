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

![](/ICP/3D/resources/2d_ICP.gif)

A considerable speed improvement can be gained by storing the target point set in a K-D tree and making a nearest neighbor search on the tree to find the correspondences. Here, we used [MRPT library](https://github.com/vioshyvo/mrpt) for KNN search.

## Least-squares Point-to-Plane ICP
In order to solve the rigid-body alignment more efficiently, we can consider the fact that, mostly, points measured in real world belong to some surface. Therefore, instead of minimizing point-to-point distances, we can instead minimize point-to-surface distance. Practically, this requires projecting the point-to-point error vector on to the associated surface normal (which is simply taking the dot product `{x_source-x_target)·x_target_normal}`. Note that, here we still need to find the point-wise correspondences between 2 sets, in order to compute point (set 1) to normal (set 2) projection.

Point-to-normal minimization breaks the SVD based closed-form solution though. So instead, a least-squares approach can be used to minimize the cost function. This requires the transformation equations to be linearized around the point of interest (source point), and the computation of the Jacobian matrices.

The algorithm looks as follows:
- Compute the normals of the target cloud
- 



![](/ICP/3D/resources/point_to_plane.png)
![](/ICP/3D/resources/tangent.png)
![](/ICP/3D/resources/equation_point_to_plane.png)




### References
https://en.wikipedia.org/wiki/Kabsch_algorithm
https://nbviewer.org/github/niosus/notebooks/blob/master/icp.ipynb#ICP-based-on-SVD
https://gfx.cs.princeton.edu/proj/iccv05_course/iccv05_icp_gr.pdf