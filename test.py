import sys

import torch
import numpy as np
import open3d

center = [0, 0, 0]
lwh = [100,100,100]
axis_angles = np.array([0, 0, 0])
rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

# import ipdb; ipdb.set_trace(context=20)
lines = np.asarray(line_set.lines)
lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

line_set.lines = open3d.utility.Vector2iVector(lines)
vis = open3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().point_size = 1.0
vis.get_render_option().background_color = np.zeros(3)

axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
vis.add_geometry(axis_pcd)

pts = open3d.geometry.PointCloud()
pts.points = open3d.utility.Vector3dVector([[1,1,1]])
vis.add_geometry(pts)

vis.add_geometry(line_set)
line_set.paint_uniform_color((0, 1, 0))
vis.run()
vis.destroy_window()
