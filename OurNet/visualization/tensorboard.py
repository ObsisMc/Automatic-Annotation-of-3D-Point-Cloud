import os
import torch
import common_utils.cfgs as Config
from torch.utils.tensorboard import SummaryWriter


class TensorBoardVis:

    def __init__(self, path=None, net=None):
        if path is None:
            path = Config.load_model_visual()
            path = os.path.join(path, net.__class__.__name__, "negative")
            if not os.path.exists(path):
                os.makedirs(path)
        self.writer = SummaryWriter(path)

    def add_scalar(self, title, value, epoch=None):
        self.writer.add_scalar(title, value, epoch)
