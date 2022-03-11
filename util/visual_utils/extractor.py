from util.visual_utils.visual_modul.calibration import Calibration
from visual_modul import open3d_vis_utils as V, io_utils as io
import open3d
import numpy as np

basepath = "../../"
pointspath = basepath + "test/training/velodyne/0000/000000.bin"
boxpath = basepath + "test/training/label/000000.txt"
calibpath = basepath + "test/training/calib/0000.txt"

calibration = Calibration(calibpath)  # used to transfer the axis to lidar's

points = io.load_points(pointspath)
boxes = io.load_boxes_from_object_txt(boxpath)
boxes = calibration.bbox_rect_to_lidar(boxes)[0].reshape(1, -1)  # gain the right boxes

pcld = open3d.geometry.PointCloud()
pcld.points = open3d.utility.Vector3dVector(points)


def extract_box(gt_boxes=np.array([[0, 0, 0, 30, 30, 30, 0]])):
    center = gt_boxes[0, 0:3]
    lwh = gt_boxes[0, 3:6]
    axis_angles = np.array([0, 0, gt_boxes[0, 6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    return box3d


# in this way can extract points
bound = extract_box(boxes)
pcld_crop = pcld.crop(bound)
print(np.asarray(pcld_crop.points))
print(boxes)


# open3d.visualization.draw_geometries([pcld_crop])
# open3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)

# need to make it orthogonal and make the origin to its center
def canonicalize(points: np.ndarray, boxes: np.ndarray):
    # translation
    center = boxes[0, 0:3].reshape(1, -1)
    points_translated = (points - center).reshape(1, -1)

    # rotation
    points_canonical = V.rotate_points_along_z(points_translated.reshape(1, -1, 3), -boxes[0, -1])
    # points_canonical = points_translated.reshape(1, -1, 3)
    return points_canonical

# if there is axis will be better
points_canonical = canonicalize(np.asarray(pcld_crop.points), boxes)[0]
pcld_crop.points = open3d.utility.Vector3dVector(points_canonical)
print(np.asarray(pcld_crop.points))

open3d.visualization.draw_geometries([pcld_crop])
