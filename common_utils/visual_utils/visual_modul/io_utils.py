import numpy as np
import open3d as o3d
import os
import logging


def load_oxst_tracking_scene(oxst_scene_path):
    """
    [lat, lon, alt, row, pitch, yaw]
    """
    oxst = []
    with open(oxst_scene_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            oxst.append(line.split(" ")[:6])
    return oxst


def load_points(pointspath):
    if pointspath.endswith(".bin"):
        return np.fromfile(pointspath, dtype=np.float32).reshape(-1, 4)
    elif pointspath.endswith(".npy"):
        points = np.load(pointspath)
        if points.shape[1] == 4:
            points = points.reshape(-1, 4)
        return points
    return None


def trans_track_txt_to_calib_format(box: str, extend=1):
    """
    @return dimension is (N,3)
    """
    box = box.rstrip('\n').rstrip(" ").split(" ")
    extend_mtx = np.array([1] * 3 + [extend] * 3 + [1]).reshape(1, -1)
    trans_box = np.array(box[13:16] + [box[12], box[10], box[11]] + [box[16]], dtype=np.float32).reshape(1, -1)
    return (trans_box * extend_mtx).reshape(1, -1)


def load_boxes_from_object_txt(boxpath):
    """
    1. the boxes should be calibrated
    2. box should be [location + l + h + w + angle]
    3. label is in kitti object format

    @return dimension is (N,3)
    """

    def parseLabel(category):
        if category == "Car":
            return 1  # green
        elif category == "Pedestrian":
            return 2  # blue
        elif category == "Cyclist":
            return 3  # yellow
        elif category == "Van":
            return 4
        elif category == "Truck":
            return 5
        return 0

    boxes = []
    labels = []
    with open(boxpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip("\n").split(" ")
            if line[0] == "DontCare":
                continue
            box = line[11:14] + [line[10], line[8], line[9]] + [line[14]]
            boxes.append(box)
            labels.append(parseLabel(line[0]))
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int)


def save_object_points(points: np.ndarray, outputroot, name):
    if not os.path.exists(outputroot):
        os.makedirs(outputroot)
    np.save(os.path.join(outputroot, name), points)


def save_object_label(label, outputroot, name):
    """
    label: [location,l,h,w,angle] float  without ""\n
    """
    if not os.path.exists(outputroot):
        os.makedirs(outputroot)
    with open(os.path.join(outputroot, name), "w") as f:
        f.write(" ".join([str(v) for v in label]))
        f.write("\n")


def logFileNotFound(path):
    logging.basicConfig(level=logging.DEBUG)
    LOG_FORMAT = "%(asctime)s - %(message)s"
    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format=LOG_FORMAT)
    logging.debug("FileNotFoundError: no %s" % path)
