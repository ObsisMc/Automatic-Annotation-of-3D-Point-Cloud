import numpy as np
import common_utils.cfgs as Config
from visual_modul import open3d_vis_utils as V, io_utils as io
from visual_modul.calibration import Calibration


def visualize_scene(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                    draw_origin=True):
    """
    it only supports single frame
    """
    V.draw_scenes(
        points=points, ref_boxes=ref_boxes, ref_labels=ref_labels, ref_scores=ref_scores, point_colors=point_colors
    )


def visualize_object(points: np.ndarray, ref_boxes=None, points2=None, colorful=True):
    V.draw_object(points, ref_boxes, points2, colorful=colorful)


def main():
    cfg = Config.load_visual("visualize_root")

    # used to transfer the coordinates to lidar's
    calibration = Calibration(cfg["calibpath"])

    # load data
    points = io.load_points(cfg["objectpath"])
    boxes, label = io.load_boxes_from_object_txt(cfg["boxpath"])
    boxes = calibration.bbox_rect_to_lidar(boxes)  # move coordinates

    # visualize
    # visualize_scene(points=points, ref_boxes=boxes, ref_labels=label)
    # visualize_object(points=points, ref_boxes=boxes[0], keep_world_coord=cfg["keep_world_coord"])
    # visualize_object(points=points)

    # show 2 points in a window
    multi_point = []
    for i in range(len(cfg["multi_points"])):
        multi_point.append(io.load_points(cfg["multi_points"][i]))
    visualize_object(points=points, points2=multi_point, colorful=False)


if __name__ == '__main__':
    main()
