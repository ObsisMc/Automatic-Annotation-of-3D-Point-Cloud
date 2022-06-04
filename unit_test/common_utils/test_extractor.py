import os.path

import pytest
import numpy as np

from common_utils.visual_utils.visual_modul.calibration import Calibration
from common_utils.visual_utils.extractor import extract_single_object
import common_utils.visual_utils.visual_modul.io_utils as io_utils


def test_extract_single_object():
    scene_points = io_utils.load_points("/home/zrh/Data/kitti/data_tracking_velodyne/0000/000000.bin")[:, :3]
    calib = Calibration("/home/zrh/Data/kitti/data_tracking_calib/training/calib/0000.txt")
    box = "0 0 Van 0 0 -1.793451 296.744956 161.752147 455.226042 292.372804 2.000000 1.823255 4.433886 -4.552284 1.858523 13.410495 -2.115488"
    gt_extracted_points = np.load(
        "/home/zrh/Data/kitti/tracking/extracted_points_canonical/0000/Van#0/points/000000.npy")
    extracted_points = extract_single_object(scene_points, calib, box, 1.3)

    diff = np.abs(gt_extracted_points - extracted_points)
    assert np.all(diff < 1e-5)


def test_extract_tracking_scene_default():
    eval_frame = "000148"
    gt_path = "/home/zrh/Data/kitti/tracking/extracted_points_entend13/0000/Van#0"
    test_path = "/home/zrh/Data/kitti/tracking/extracted_points_test_default/0000/Van#0"

    gt_points = io_utils.load_points(os.path.join(gt_path, "points", eval_frame + ".npy"))[:, :3]
    test_points = io_utils.load_points(os.path.join(test_path, "points", eval_frame + ".npy"))[:, :3]
    assert np.all(gt_points - test_points < 1e-5)

    with open(os.path.join(gt_path, "labels", eval_frame + ".txt"), "r") as f:
        gt_label = f.readlines()
    with open(os.path.join(test_path, "labels", eval_frame + ".txt"), "r") as f:
        test_label = f.readlines()

    gt_label = np.array(gt_label[0].split(" "), dtype=np.float)
    test_label = np.array(test_label[0].split(" "), dtype=np.float)
    assert np.all(gt_label - test_label < 1e-5)


if __name__ == "__main__":
    pytest.main(["/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech"])
