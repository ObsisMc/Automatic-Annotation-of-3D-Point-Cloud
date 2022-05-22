import pytest
import numpy as np

from common_utils.visual_utils.visual_modul.calibration import Calibration
from common_utils.visual_utils.extractor import extract_single_object
import common_utils.visual_utils.visual_modul.io_utils as io_utils


def test_extract_single_object():
    scene_points = io_utils.load_points("/home/zrh/Data/kitti/data_tracking_velodyne/0000/000000.bin")[:, :3]
    calib = Calibration("/home/zrh/Data/kitti/data_tracking_calib/training/calib/0000.txt")
    box = "0 0 Van 0 0 -1.793451 296.744956 161.752147 455.226042 292.372804 2.000000 1.823255 4.433886 -4.552284 1.858523 13.410495 -2.115488"
    gt_extracted_points = np.load("/home/zrh/Data/kitti/tracking/extracted_points/0000/Van#0/points/000000.npy")
    extracted_points = extract_single_object(scene_points, calib, box, 1.3)

    diff = np.abs(gt_extracted_points - extracted_points)
    assert np.all(diff < 1e-5)


if __name__ == "__main__":
    pytest.main(["/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech"])
