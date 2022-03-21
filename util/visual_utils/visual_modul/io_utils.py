import numpy as np
import open3d as o3d
import os


def load_points(pointspath):
    if pointspath.endswith(".bin"):
        points = np.fromfile(pointspath, dtype=np.float32).reshape(-1, 4)[:, :3]
    elif pointspath.endswith(".npy"):
        points = np.load(pointspath)
        if points.shape[1] == 4:
            points = points.reshape(-1, 4)
        points = points[:, :3]
    return points


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


def save_object_points(points: np.ndarray, outputroot, name):
    if not os.path.exists(outputroot):
        os.makedirs(outputroot)
    np.save(os.path.join(outputroot, name), points)


def save_object_label(label, outputroot, name):
    """
    label: [l,h,w,angle] float  without ""\n
    """
    if not os.path.exists(outputroot):
        os.makedirs(outputroot)
    with open(os.path.join(outputroot, name), "w") as f:
        f.write("{} {} {} {}".format(*label))
        f.write("\n")
