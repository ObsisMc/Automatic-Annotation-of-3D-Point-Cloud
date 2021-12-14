import numpy as np
import os
from scipy.spatial import Delaunay


def getFrameNumber(path):
    return len(os.listdir(path))


def load_point_clouds(path):
    '''
    Args:
        path: string, containing point clouds with extension .bin
    Returns:
        points: (N, 3)
    '''
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def load_3d_boxes(path, dilate=1):
    '''
    Load 3d bounding boxes from label
    Args:
        path: string, containing the label of 3d bounding boxes with extension .txt
        category: category to be loaded.
                {'Car','Van','Truck','Pedestrian','Person_sitting','Cyclist','Tram'}
    Returns:
        boxes: (M, 7), [x, y, z, dx, dy, dz, heading]
    '''

    def parse_line(line):
        label = line.strip().split(' ')
        framei = [label[0]]
        id = [label[1]]
        cat = label[2]
        h, w, l = [label[10]], [label[11]], [label[12]]
        loc = label[13:16]
        heading = [label[16]]
        boxes = framei + id + loc + l + h + w + heading
        # 为啥对不上h,w,l???
        return np.array(boxes, dtype=np.float32) * np.array([1] * 5 + [dilate, dilate, dilate * 1.2] + [1]), cat

    with open(path, 'r') as f:
        lines = f.readlines()

    boxes = []
    cates = []
    for line in lines:
        box, cat = parse_line(line)
        boxes.append(box)
        cates.append(cat)

    assert len(boxes) != 0

    return np.array(boxes), cates


def label_rect_to_lidar(label):
    '''
    Transform the bbox from camera system to the lidar system.
    Args:
        label: [N, 7], [x, y, z, dx, dy, dz, heading]
    Returns:
        label_lidar: [N, 7]
    '''
    loc, dims, rots = lable[:, :3], lable[:, 3:6], lable[:, 6]
    loc_lidar = calib.rect_to_lidar(loc)
    l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    loc_lidar[:, 2] += h[:, 0] / 2
    label_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    return label_lidar


def is_within_3d_box(points, corners3d):
    """
    Check whether a point is within bbox
    Args:
        points:  (N, 3)
        corners3d: (M, 8, 3), corners of M bboxes.
    Returns:
        flag: (M, N), bool
    """
    num_objects = len(corners3d)
    flag = []
    for i in range(num_objects):
        hull = Delaunay(corners3d[i])
        flag.append(hull.find_simplex(points) >= 0)
    flag = np.stack(flag, axis=0)
    return flag


def lidar_to_shapenet(points, box):
    '''
    Transform the coordinates from kitti(lidar system) to shapenet.
    kitti: x->forward, z->up
    shapenet: x->forward, y->up
    Args:
        points: (N, 3), points within bbox
        box: (7), box parameters
    Returns:
        points_new: (N, 3), points within bbox
        box_new: (7), box parameters
    '''
    rot = np.array(-90 / 180 * np.pi).reshape(1)
    points_new = rotate_points_along_x(points, rot)

    box_new = box.copy()
    box_new[4], box_new[5] = box_new[5], box_new[4]  # 只换dy 和 dz，不用改中心（因为已经把中心移到零点了）
    # box_new[6] = rot_y  # 这是以z为轴转，但实际要以y为轴
    return points_new.squeeze(), box_new


def points_to_canonical(points, box, oriangle=False):
    '''
    Transform points within bbox to a canonical pose and normalized scale
    Args:
        points: (N, 3), points within bbox
        box: (7), box parameters
    Returns:
        points_canonical: (N, 3)
    '''
    center = box[:3].reshape(1, 3)
    rot = -box[-1].reshape(1)  # 将点转正
    points_centered = (points - center).reshape(1, -1, 3)
    if not oriangle:
        points_centered_rot = rotate_points_along_z(points_centered, rot)
    else:
        points_centered_rot = points_centered
    scale = (1 / np.abs(box[3:6]).max()) * 0.999999
    points_canonical = points_centered_rot * scale

    box_canonical = box.copy()
    box_canonical[:3] = 0  # 中心移到零点
    if not oriangle:
        box_canonical[-1] = 0  # 沿y的角度为0
        box_canonical = box_canonical * scale
    else:
        box_canonical[:6] = box_canonical[:6] * scale

    return points_canonical.squeeze(), box_canonical


def boxes_to_corners_3d(boxes3d, oriangle=False):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center # heading is angle
    Returns:
        corners3d: (N, 8, 3)
    """
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ]) / 2

    corners3d = boxes3d[:, None, 3:6] * template[None, :, :]  # padding for multiply
    if oriangle:
        corners3d = rotate_points_along_y(corners3d, boxes3d[:, 6]).reshape(-1, 8, 3)
    else:
        corners3d = rotate_points_along_z(corners3d, boxes3d[:, 6]).reshape(-1, 8,
                                                                            3)  # rotate corners3d along z angle
    corners3d += boxes3d[:, None, 0:3]
    return corners3d


def rotate_points_along_x(points, angle):
    """
    Args:
        points: (B, N, 3)
        angle: (B), angle along x-axis
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    ones = np.ones_like(angle, dtype=np.float32)
    zeros = np.zeros_like(angle, dtype=np.float32)
    # 沿x轴旋转
    rot_matrix = np.stack((
        ones, zeros, zeros,
        zeros, cosa, sina,
        zeros, -sina, cosa
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points, rot_matrix)
    return points_rot


def rotate_points_along_y(points, angle):
    cosa = np.cos(angle)
    sina = np.sin(angle)
    ones = np.ones_like(angle, dtype=np.float32)
    zeros = np.zeros_like(angle, dtype=np.float32)
    # 沿x轴旋转
    rot_matrix = np.stack((
        cosa, zeros, -sina,
        zeros, ones, zeros,
        sina, zeros, cosa
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points, rot_matrix)
    return points_rot


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
    rot_matrix = np.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points, rot_matrix)
    return points_rot


def write_bboxes(bboxes, save_path):
    if not os.path.exists(save_path[0]):
        os.makedirs(save_path[0])
    np.save(os.path.join(save_path[0], save_path[1]), bboxes)
    return


def write_points(points, save_path):
    if not os.path.exists(save_path[0]):
        os.makedirs(save_path[0])
    np.save(os.path.join(save_path[0], save_path[1]), points)
    return
