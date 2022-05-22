import pytest
import numpy as np
from common_utils.visual_utils.visual_modul.calibration import Calibration
from common_utils.visual_utils.extractor import extract_single_object


def test_extract_single_object():
    scene_points = np.load('')
    calib = Calibration("/home/zrh/Data/kitti/data_tracking_calib/training/calib/0000.txt")
    box = ""
    gt_extracted_points = np.load("")
    extract_single_object(scene_points, calib, box, 1.3)
