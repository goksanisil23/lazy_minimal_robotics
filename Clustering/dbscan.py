import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

pcd_file = "/home/goksan/Downloads/TLS_kitchen.ply"
pcd = o3d.io.read_point_cloud(pcd_file)

# Estimate normals for better visualization
# for each point, compute the normal by looking at a search radius of 0.1m
# and maximum of 16 neighbor points to calculate the principal axes
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.1, max_nn=16), fast_normal_computation=True)

plane_segments = {}
plane_models = {}
MAX_NUM_PLANES = 20
PLANE_DIST_THRES = 0.01

rest = pcd

for i in range(MAX_NUM_PLANES):
    colors = plt.get_cmap("tab20")(i)

    plane_models[i], inliers = rest.segment_plane(
        distance_threshold=PLANE_DIST_THRES, ransac_n=3, num_iterations=1000)
    plane_segments[i] = rest.select_by_index(inliers)

    # within this plane, run dbscan clustering to refine
    labels_per_pt = np.array(plane_segments[i].cluster_dbscan(
        eps=PLANE_DIST_THRES*10, min_points=10))
    # find number of elements per each unique label
    num_pts_per_label = [len(np.where(labels_per_pt == j)[0])
                         for j in np.unique(labels_per_pt)]
    # find the label with the largest number of points
    best_label = int(np.unique(labels_per_pt)[np.where(
        num_pts_per_label == np.max(num_pts_per_label))[0]])

    # re-assign the plane points after dbscan reclustering
    refined_inlier_idxs = list(np.where(labels_per_pt == best_label)[0])
    refined_outlier_idxs = list(np.where(labels_per_pt != best_label)[0])

    rest = rest.select_by_index(
        inliers, invert=True) + plane_segments[i].select_by_index(refined_outlier_idxs)
    plane_segments[i] = plane_segments[i].select_by_index(refined_inlier_idxs)

    plane_segments[i].paint_uniform_color(list(colors[:3]))
    print("pass", i, "/", MAX_NUM_PLANES, "done.")

# CLuster the remaining points
# labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")

# # colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
# # colors[labels < 0] = 0
# # rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries(
    [plane_segments[i] for i in range(MAX_NUM_PLANES)]+[rest])
