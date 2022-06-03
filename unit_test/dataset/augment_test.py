import numpy as np

from OurNet.dataset.dataset_utils.Augmentor import Augmentor
import OurNet.dataset.dataset_utils.utils as utils


def test_augment():
    augmentor = Augmentor()
    points = np.zeros((0, 3))
    points = utils.rotate_points_along_z(points, 1)
    augmentor.guassianAug(points=points, conf=1)
