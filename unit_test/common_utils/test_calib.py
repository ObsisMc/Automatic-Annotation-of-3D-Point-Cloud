from common_utils.visual_utils.visual_modul.calibration import Calibration
import numpy as np


def test_calib():
    print()
    extend = 1.3
    calibration = Calibration("/home/zrh/Data/kitti/data_tracking_calib/training/calib/0000.txt")
    lidar_label = "13.701017767989608 4.57136428789947 -0.4423585236442056 5.764051866531372 2.3702314257621766 2.6 0.5446917255732675"
    bbox_label = "2.000000 1.823255 4.433886 -4.552284 1.858523 13.410495 -2.115488"

    lidar_list = np.array(lidar_label.split(" "), dtype=np.float)

    test_bbox = calibration.lidar_to_bbox_rect(lidar_list.reshape(1, -1), extend, True)

    bbox_list = np.array(bbox_label.split(" "), dtype=np.float).reshape(1, -1)
    print(test_bbox)
    print(bbox_list)
    assert np.all(np.abs(test_bbox - bbox_list) < 1e-2)
