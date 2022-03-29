import numpy as np

from visual_modul import open3d_vis_utils as V, io_utils as io
from visual_modul.calibration import Calibration

basepath = ""
pointspath = basepath + "/home/zrh/Data/kitti/data_tracking_velodyne/0000/000000.bin"
objectpath = basepath + "/home/zrh/Data/kitti/tracking/extracted_points/0000/Van#0/points/000000.npy"
objectpath2 = basepath + "/home/zrh/Data/kitti/tracking/extracted_points/0000/Van#0/points/000001.npy"
boxpath = basepath + "/home/zrh/Data/kitti/data_tracking_label_2/training/label_2/0000.txt"  # label's path (in object format)
calibpath = basepath + "/home/zrh/Data/kitti/data_tracking_calib/training/calib/0000.txt"


def visualize_scene(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                    draw_origin=True):
    """
    it only supports single frame
    """
    V.draw_scenes(
        points=points, ref_boxes=ref_boxes
    )


def visualize_object(points: np.ndarray, ref_boxes=None,points2 = None):
    V.draw_object(points, ref_boxes, points2)


if __name__ == '__main__':
    # used to transfer the coordinates to lidar's
    calibration = Calibration(calibpath)

    # load data
    points = io.load_points(objectpath2)
    boxes = io.load_boxes_from_object_txt(boxpath)
    boxes = calibration.bbox_rect_to_lidar(boxes)  # move coordinates

    # visualize
    # visualize_scene(points=points, ref_boxes=boxes)
    # visualize_object(points=points, ref_boxes=boxes[0])
    visualize_object(points=points)

    # show 2 points in a window
    # points2 = io.load_points(objectpath2)
    # visualize_object(points=points, points2=points2)
