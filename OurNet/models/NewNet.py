import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import PointNet


class NewNet(torch.nn):
    def __init__(self, k):
        self.pointfeat = PointNet.PointNetfeat()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, k)
        self.dropout = nn.Dropout(0.2)
        pass

    def forward(self, x1, x2):
        x1, _, _ = self.pointfeat(x1)
        x2, _, _ = self.pointfeat(x2)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.dropout(self.fc1(x)))
        x = F.leaky_relu(self.dropout(self.fc2(x)))
        x = F.leaky_relu(self.dropout(self.fc3(x)))
        x = self.fc4(x)

        return x
