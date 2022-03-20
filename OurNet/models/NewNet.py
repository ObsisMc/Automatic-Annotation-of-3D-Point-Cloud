import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import PointNet


class NewNet(torch.nn):
    def __init__(self):
        self.pointnet = PointNet.PointNetfeat()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        pass
