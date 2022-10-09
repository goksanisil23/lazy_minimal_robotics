import laspy as lp
import numpy as np
import open3d as o3d

input_path = "/home/goksan/Downloads/"
dataname = "2021_heerlen_table.las"

VOXEL_FACTOR = 0.005  # ratio of largest bounding box edge to voxel edge

pointcloud = lp.read(input_path+dataname)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(
    np.vstack((pointcloud.x, pointcloud.y, pointcloud.z)).transpose())
pcd.colors = o3d.utility.Vector3dVector(np.vstack(
    (pointcloud.red, pointcloud.green, pointcloud.blue)).transpose()/65535)

# voxel size based on the bounding box of the 3d asset
vox_size = round(max(pcd.get_max_bound()-pcd.get_min_bound())*VOXEL_FACTOR, 4)

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    pcd, voxel_size=vox_size)

# Create a triangle mesh for exporting from the voxelized asset
vox_mesh = o3d.geometry.TriangleMesh()
for vox in voxel_grid.get_voxels():
    cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
    cube.paint_uniform_color(vox.color)
    cube.translate(vox.grid_index, relative=False)
    vox_mesh += cube

vox_mesh.merge_close_vertices(0.0000001)
o3d.io.write_triangle_mesh(input_path + "voxel_mesh_h.ply", vox_mesh)
