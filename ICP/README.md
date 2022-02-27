## Iterative Closest Point (ICP)
ICP is a method to solve the rigid-transform (rotation + translation) problem between 2 sets of points. These 2 sets can be both partially or fully overlapping.
The "iterative" aspect of ICP is due to the fact that point-wise correspondences between 2 sets of data are (mostly) unknown. 

If the correspondences were known, finding the optimal rotation and translation between those 2 sets can be solved by using Kabsch algorithmm **in 1 step**. 2 important factors here are:
- Covariance matrix of 2 sets: Defined by the inner product of 2 sets of **origo-centered points**: (3xN) * (Nx3) = (3x3)
- Singular Value Decomposition: U*V^T of SVD is intuitively the direction of the rotational difference between 2 sets of data.
```c++
Eigen::Matrix3d H = input_orig * target_orig.transpose(); // H = 3x3
Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
Eigen::Matrix3d R = svd.matrixV() * I * svd.matrixU().transpose();
```

But since we don't know the correspondences, ICP makes certain assumptions. One of the simplest flavors of ICP takes closest points between 2 sets as correspondences. For certain number of iterations, this assumption is most probably wrong. (In the example below, this can be seen by 1-to-many correspondences). However, if 2 sets were reasonably close, this approach gets us closer to the actual correspondences, and therefore to the actual transformation. 

- Correlation matrix H is computed between the estimated correspondences between the input and target point sets
- ICP stops when the correspondence from the previous iteration doesn't change anymore.

![](/ICP/simple/2d.gif)






### References
https://en.wikipedia.org/wiki/Kabsch_algorithm
https://nbviewer.org/github/niosus/notebooks/blob/master/icp.ipynb#ICP-based-on-SVD