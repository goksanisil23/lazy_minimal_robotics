import random
import sys
import numpy as np
import open3d as o3d
from mpl_toolkits import mplot3d
import plotly.express as px

from sklearn.neighbors import KDTree

############ GLOBALS ############

# dataset = "/home/goksan/Downloads/the_adas_lidar.xyz"
dataset = "/home/goksan/Downloads/the_researcher_desk.xyz"
# dataset = "/home/goksan/Downloads/the_playground.xyz"

############ METHODS ############


def ransac_plane(xyz, threshold=0.05, iterations=1000):
    best_inliers = []
    n_points = len(xyz)
    best_plane_eq = []
    i_iter = 1

    # Note that setting iterations like this does not guarantee that we will find the best 3 points in this dataset that represent the most obvious plane
    # It only determines how long we should be searching for, by picking random points each time
    while i_iter < iterations:
        # 3 random points A,B,C
        idx_samples = random.sample(range(len(xyz)), 3)
        pts = xyz[idx_samples]

        # Cross product of ABxBC defines the normal of the plane that holds these 3 points
        vecA = pts[1] - pts[0]
        vecB = pts[2] - pts[0]
        normal = np.cross(vecA, vecB)
        # normalized normal vector coefficients are a,b,c, in plane equation
        a, b, c = normal/np.linalg.norm(normal)
        # Find d by using one of the A,B,C points that satisfies the ax+by+cz+d=0 equation
        # Note that d is found with non-normalized Normal vector since it's up to scale
        d = -np.sum(normal*pts[0])
        # Find the (shortest) distance of each point in the pointcloud to this plane
        # D = (a*x1 + b*y1 + c*z1 + d) / sqrt(a²+b²+c²)
        distancesToPlane = (a*xyz[:, 0] + b*xyz[:, 1] + c *
                            xyz[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        idx_candidates = np.where(np.abs(distancesToPlane) <= threshold)[0]

        if (len(idx_candidates) > len(best_inliers)):
            best_plane_eq = [a, b, c, d]
            best_inliers = idx_candidates
        i_iter += 1

    return best_plane_eq, best_inliers


def ransac_plane_with_adaptive_iteration(xyz, threshold=0.05):
    best_inliers = []
    n_points = len(xyz)
    best_plane_eq = []
    i_iter = 1
    inlier_ratio = 0.0  # start with worst case scenario
    iteration_limit_N = sys.maxsize
    prob = 0.99
    min_samples = 3  # since we're working with 3 points on a plane

    while i_iter < iteration_limit_N:
        # 3 random points A,B,C
        idx_samples = random.sample(range(len(xyz)), 3)
        pts = xyz[idx_samples]

        # Cross product of ABxBC defines the normal of the plane that holds these 3 points
        vecA = pts[1] - pts[0]
        vecB = pts[2] - pts[0]
        normal = np.cross(vecA, vecB)
        # normalized normal vector coefficients are a,b,c, in plane equation
        a, b, c = normal/np.linalg.norm(normal)
        # Find d by using one of the A,B,C points that satisfies the ax+by+cz+d=0 equation
        # Note that d is found with non-normalized Normal vector since it's up to scale
        d = -np.sum(normal*pts[0])
        # Find the (shortest) distance of each point in the pointcloud to this plane
        # D = (a*x1 + b*y1 + c*z1 + d) / sqrt(a²+b²+c²)
        distancesToPlane = (a*xyz[:, 0] + b*xyz[:, 1] + c *
                            xyz[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        idx_candidates = np.where(np.abs(distancesToPlane) <= threshold)[0]

        if (len(idx_candidates) > len(best_inliers)):
            best_plane_eq = [a, b, c, d]
            best_inliers = idx_candidates
            # Update the max iteration
            inlier_ratio = len(best_inliers) / len(xyz)
            iteration_limit_N = np.log(
                1.0-prob) / np.log(1.0-(inlier_ratio)**min_samples)
            print("iteration limit updated to: {}".format(iteration_limit_N))

        i_iter += 1

    return best_plane_eq, best_inliers


############ MAIN ############
pcd = np.loadtxt(dataset, skiprows=1)

xyz = pcd[:, :3]
rgb = pcd[:, 3:6]

print("num points in the cloud: {}".format(len(xyz)))

iterations = 1000
num_nearest_neighbors = 15

tree = KDTree(np.array(xyz), leaf_size=2)
nearest_dist, nearest_ind = tree.query(xyz, k=num_nearest_neighbors)
avg_nn_dist = np.mean(nearest_dist[:, 1:], axis=0)
threshold = np.mean(avg_nn_dist)
print("threshold from avg. nn. dist.: {}".format(threshold))

# # don't take 0th column since thats the point itself
# mean_nearest_dist = np.mean(nearest_dist[:, 1:], axis=0)

# plane_eq, idx_inliers = ransac_plane(xyz, threshold, iterations)
plane_eq, idx_inliers = ransac_plane_with_adaptive_iteration(xyz, threshold)
inlier_pts = xyz[idx_inliers]

mask = np.ones(len(xyz), dtype=bool)
mask[idx_inliers] = False
outlier_pts = xyz[mask]

# Visualize
print("drawing ...")
o3d_inlier_pts = o3d.geometry.PointCloud()
o3d_inlier_pts.points = o3d.utility.Vector3dVector(inlier_pts)
o3d_inlier_pts.colors = o3d.utility.Vector3dVector(
    np.tile(np.array([0, 1, 0]), (len(inlier_pts), 1)))
o3d_outlier_pts = o3d.geometry.PointCloud()
o3d_outlier_pts.points = o3d.utility.Vector3dVector(outlier_pts)
o3d_outlier_pts.colors = o3d.utility.Vector3dVector(
    np.tile(np.array([0, 0, 1]), (len(outlier_pts), 1)))
o3d.visualization.draw_geometries([o3d_outlier_pts, o3d_inlier_pts])

##################################### basic kd-tree usage ###################################
# import numpy as np
# from sklearn.neighbors import KDTree
# rng = np.random.RandomState(0)
# # X = rng.random_sample((10, 3))  # 10 points in 3 dimensions
# X = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [
#              6, 0, 0], [7, 0, 0], [8, 0, 0], [9, 0, 0], [10, 0, 0]])
# tree = KDTree(X, leaf_size=4)

# target_idx = 5
# target_pt = X[target_idx, :].reshape(1, -1)
# print("find 3 closest neighbors to pt in idx {}: {}".format(target_idx, target_pt))

# dist, ind = tree.query(target_pt, k=3)

# # indices of 3 closest neighbors
# print("closest indices to target: {}".format(ind))

# print("distances: {}".format(dist))  # distances to 3 closest neighbors
