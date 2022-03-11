from visual_modul import open3d_vis_utils as V
from visual_modul.calibration import Calibration
import numpy as np

basepath = "../../"
pointspath = basepath + "test/training/velodyne/0000/000000.bin"
boxpath = basepath + "test/training/label/000000.txt"
calibpath = basepath + "test/training/calib/0000.txt"


def load_points():
    points = np.fromfile(pointspath, dtype=np.float32).reshape(-1, 4)
    return points[:, :]


def load_boxes_from_object_txt():
    """
    1. the boxes should be calibrated
    2. box should be [location + l + h + w + angle]
    """
    boxes = []
    with open(boxpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip("\n").split(" ")
            if line[0] == "DontCare":
                continue
            box = line[11:14] + [line[10], line[8], line[9]] + [line[14]]
            boxes.append(box)
    return np.array(boxes, dtype=np.float32)


# V.draw_scenes(
#     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
#     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
# )


if __name__ == '__main__':
    calibration = Calibration(calibpath)

    points = load_points()
    boxes = load_boxes_from_object_txt()
    boxes = calibration.bbox_rect_to_lidar(boxes)
    V.draw_scenes(
        points=points, ref_boxes=boxes
    )
