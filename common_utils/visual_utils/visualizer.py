import numpy as np
import os
import common_utils.cfgs as Config
from visual_modul import open3d_vis_utils as V, io_utils as io
from visual_modul.calibration import Calibration
from visual_modul.oxst_projector import OxstProjector


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


def show_continuous_objects(cfg, oxst_projector: OxstProjector, oxst: list):
    root = cfg["multi_points"][0].rsplit("/", 1)[0]
    begin = int(cfg["multi_points"][0].rsplit("/", 1)[1].rstrip(".npy"))
    end = int(cfg["multi_points"][-1].rsplit("/", 1)[1].rstrip(".npy"))
    multi_point = []

    base_position = None
    for i in range(begin, end + 1):
        oxst_projector.init_oxst(oxst[i])

        path = os.path.join(root, "{:06d}.npy".format(i))
        if os.path.exists(path):
            points = io.load_points(path)
            points = oxst_projector.lidar_to_pose(points)
            multi_point.append(points)
    visualize_object(points=None, points2=multi_point, colorful=False)


def show_continuous_objects_without_adjust(cfg):
    root = cfg["multi_points"][0].rsplit("/", 1)[0]
    begin = int(cfg["multi_points"][0].rsplit("/", 1)[1].rstrip(".npy"))
    end = int(cfg["multi_points"][-1].rsplit("/", 1)[1].rstrip(".npy"))
    multi_point = []
    for i in range(begin, end + 1):

        path = os.path.join(root, "{:06d}.npy".format(i))
        if os.path.exists(path):
            points = io.load_points(path)
            multi_point.append(points)
            print(points[:3, :])
    visualize_object(points=None, points2=multi_point, colorful=True)


def main():
    cfg = Config.load_visual("visualize_root")

    # used to transfer the coordinates to lidar's
    calibration = Calibration(cfg["calibpath"])

    # get oxst pose
    oxst_projector = OxstProjector()
    scene_frame = 0
    oxsts = io.load_oxst_tracking_scene("/home/zrh/Data/kitti/data_tracking_oxts/training/oxts/0000.txt")

    # load data
    points = io.load_points(cfg["objectpath"])
    boxes, label = io.load_boxes_from_object_txt(cfg["boxpath"])
    boxes = calibration.bbox_rect_to_lidar(boxes)  # move coordinates

    # visualize scene and its label
    # visualize_scene(points=points, ref_boxes=boxes, ref_labels=label)

    # visualize a object in a scene
    # visualize_object(points=points, ref_boxes=boxes[0], keep_world_coord=cfg["keep_world_coord"])

    # visualize a extracted object
    visualize_object(points=points[:, :3])

    # show many objects in a window
    # multi_point = []
    # for i in range(len(cfg["multi_points"])):
    #     multi_point.append(io.load_points(cfg["multi_points"][i]))
    # visualize_object(points=points, points2=multi_point, colorful=False)

    # show continuous objects in a window
    # show_continuous_objects(cfg, oxst_projector, oxsts)

    # show continuous objects in lidar coordinates
    # show_continuous_objects_without_adjust(cfg)

    # show many scenes in a window
    # multi_point = []
    # for i in range(len(cfg["multi_scenes"])):
    #     multi_point.append(io.load_points(cfg["multi_scenes"][i])[:,:3])
    # visualize_object(points=points[:,:3], points2=multi_point, colorful=False)


if __name__ == '__main__':
    main()
