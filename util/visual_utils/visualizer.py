import numpy as np

from visual_modul import open3d_vis_utils as V, io_utils as io
from visual_modul.calibration import Calibration

basepath = "./"
pointspath = basepath + "demo_data/training/velodyne/0000/000000.bin"
boxpath = basepath + "demo_data/training/label/000000.txt"  # label's path (in object format)
calibpath = basepath + "demo_data/training/calib/0000.txt"


def visualize_scene(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                    draw_origin=True):
    """
    it only supports single frame
    """
    V.draw_scenes(
        points=points, ref_boxes=ref_boxes
    )


def visualize_object(points: np.ndarray, ref_boxes: np.array):
    V.draw_object(points, ref_boxes)


if __name__ == '__main__':
    # used to transfer the coordinates to lidar's
    calibration = Calibration(calibpath)

    # load data
    points = io.load_points(pointspath)
    boxes = io.load_boxes_from_object_txt(boxpath)
    boxes = calibration.bbox_rect_to_lidar(boxes)  # move coordinates

    # visualize
    # visualize_scene(points=points, ref_boxes=boxes)
    visualize_object(points=points, ref_boxes=boxes[0])
