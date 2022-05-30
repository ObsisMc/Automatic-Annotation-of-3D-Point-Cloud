from OurNet.dataset.dataset_utils.Sampler import Sampler
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
