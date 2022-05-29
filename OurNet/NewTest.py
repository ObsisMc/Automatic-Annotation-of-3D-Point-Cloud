import torch.utils.data
from OurNet.models.model_utils import io_utils

from dataset.NewDataSet import NewDataSet
from NewModel import NewModel

device = "cuda:%d" % 0 if torch.cuda.is_available() else "cpu"


def test(points1, points2):
    dataset = NewDataSet(io_utils.getDataSetPath())
    points1, points2 = dataset.sampler.fps(points1), dataset.sampler.fps(points2)
    net = NewModel()
    # net.load_state_dict()
    net.to(device)
    output = net(points1, points2)
    return output

