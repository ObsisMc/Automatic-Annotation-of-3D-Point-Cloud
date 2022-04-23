import os
import torch
import util.cfgs as Config
from torch.utils.tensorboard import SummaryWriter


class TensorBoardVis():

    def __init__(self, path=None):
        if path is None:
            path = Config.load_model_visual()
        self.writer = SummaryWriter(path)

    def add_scalar(self, title, value, epoch=None):
        self.writer.add_scalar(title, value, epoch)
