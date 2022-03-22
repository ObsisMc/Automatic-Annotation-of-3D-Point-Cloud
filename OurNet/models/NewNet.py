import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from PointNet import PointNetfeat, SimplePointNet


class NewNet(nn.Module):
    def __init__(self):
        super(NewNet, self).__init__()
        self.pointfeat = SimplePointNet()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_refine = nn.Linear(256, 5)  # dx, dy, dz, d\theta, confidence
        self.dropout = nn.Dropout(0.2)
        pass

    def forward(self, x1, x2):
        # x1, _, _ = self.pointfeat(x1)
        # x2, _, _ = self.pointfeat(x2)
        x1 = self.pointfeat(x1)
        x2 = self.pointfeat(x2)
        x = torch.cat((x1, x2), 1)  # cat the dimension and other dimension should be the same
        x = F.leaky_relu(self.dropout(self.fc1(x)))
        x = F.leaky_relu(self.dropout(self.fc2(x)))
        x = F.leaky_relu(self.dropout(self.fc3(x)))
        x = self.fc_refine(x)

        return x[:, 3], x[:, 3], x[:, 4]
