"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],  # white
    [0, 1, 0],  # green
    [0, 1, 1],  # blue
    [1, 1, 0],  # yellow
    [1, 0, 0],  # red
    [1, 0, 1],  # purple
]


# draw a scene
def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num + 1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    box3d = extract_box(gt_boxes)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def extract_box(gt_boxes, canonical=False):
    center = gt_boxes[0:3] if not canonical else [0, 0, 0]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    return box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


# extract a single object

def extract_object(points: np.ndarray, box: np.array, keep_world_coord=False):
    """
    input:
    1. box: should be [location, l, h, w, angle]
    2. points: (N, 3) open3d.utility.Vector3dVector cannot handle refraction
    output:
    1. points_canonical: np.ndarray
    2. pcld_crop: open3d.geometry.PointCloud
    """
    # init points
    pcld = open3d.geometry.PointCloud()
    pcld.points = open3d.utility.Vector3dVector(points[:, :3])

    # extract points in box
    bound = extract_box(box)
    pcld_crop = pcld.crop(bound)

    # translate coordinates and rotate the points to be orthogonal
    points_canonical = np.asarray(pcld_crop.points)
    if not keep_world_coord:
        points_canonical = canonicalize(points_canonical, box)[0]
        pcld_crop.points = open3d.utility.Vector3dVector(points_canonical)

    # get a canonical box
    # linebox = extract_box(box, True)
    # line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(linebox)
    # lines = np.asarray(line_set.lines)
    # lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    # line_set.lines = open3d.utility.Vector2iVector(lines)
    # line_set.paint_uniform_color((1, 0, 0))
    line_set = None

    return points_canonical, pcld_crop, line_set


def draw_object(points: np.ndarray, box=None, multi_points=None, keep_world_coord=False):
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    if box is not None:
        _, pcld_crop, line_set = extract_object(points, box, keep_world_coord)
    else:
        if multi_points is not None:
            for idx, pts in enumerate(multi_points):
                pcd_tmp = open3d.geometry.PointCloud()
                pcd_tmp.points = open3d.utility.Vector3dVector(pts)
                pcd_tmp.paint_uniform_color([1, 0, 0])  # idx should less than 3
                vis.add_geometry(pcd_tmp)
        # points, _ = guassianArgu(points)
        pcld_crop = open3d.geometry.PointCloud()
        pcld_crop.points = open3d.utility.Vector3dVector(points)

    # visualize
    pcld_crop.paint_uniform_color([1, 0, 0])  # [1,0,0] is red
    vis.add_geometry(pcld_crop)

    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])  # red is x, green is y
    vis.add_geometry(axis_pcd)

    vis.run()
    vis.destroy_window()

    # another way
    # open3d.visualization.draw_geometries([pcld_crop])


def canonicalize(points: np.ndarray, boxes: np.array):
    # translation
    center = boxes[0:3].reshape(1, -1)
    points_translated = (points - center).reshape(1, -1)

    # rotation
    points_canonical = rotate_points_along_z(points_translated.reshape(1, -1, 3), -boxes[-1])
    # points_canonical = points_translated.reshape(1, -1, 3)
    return points_canonical


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    ones = np.ones_like(angle, dtype=np.float32)
    zeros = np.zeros_like(angle, dtype=np.float32)
    rot_matrix = np.array(
        [[cosa, sina, zeros],
         [-sina, cosa, zeros],
         [zeros, zeros, ones]]
    )
    points_rot = points @ rot_matrix
    return points_rot
