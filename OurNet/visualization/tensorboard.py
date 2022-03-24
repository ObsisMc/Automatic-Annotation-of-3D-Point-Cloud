import os
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter


class TensorBoardVis():

    def __init__(self, path=None):
        if path is None:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"), encoding="utf-8") as f:
                path = yaml.load(f, Loader=yaml.FullLoader)["runs_root"]
        self.writer = SummaryWriter(path)

    def add_scalar(self, title, value, epoch=None):
        self.writer.add_scalar(title, value, epoch)
