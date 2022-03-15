import numpy as np
import open3d as o3d
import os


def load_points(pointspath):
    points = np.fromfile(pointspath, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def load_boxes_from_object_txt(boxpath):
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



def save_object(points: np.ndarray, outputroot, name):
    if not os.path.exists(outputroot):
        os.makedirs(outputroot)
    np.save(os.path.join(outputroot, name), points)

