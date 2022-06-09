from OurNet.dataset.dataset_utils.Sampler import Sampler
import common_utils.visual_utils.visual_modul.open3d_vis_utils as V
import numpy as np
import copy


def test_fps():
    source_path = "/home/zrh/Data/kitti/tracking/extracted_points_entend13/0000/Van#0/points/000000.npy"
    sampler = Sampler()

    source = np.load(source_path)

    tmp = copy.deepcopy(source)
    source_fps1 = sampler.fps(tmp)

    tmp = copy.deepcopy(source)
    source_fps2 = sampler.fps(tmp)

    assert np.all(source_fps1 - source_fps2 < 1e-5)


def test_filter():
    point_path = "."
    source_path = point_path + "/000100.npy"
    sampler = Sampler()

    source = sampler.fps(np.load(source_path))
    V.draw_object(source)
    pass
