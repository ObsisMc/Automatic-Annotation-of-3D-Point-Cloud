import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


class PointNetCls(nn.Module):
    def __init__(self, k=4):
        super(PointNetCls, self).__init__()
        # self.feat = PointNetfeat()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x1, trans1, trans_feat1 = self.feat(x1)
        # x2, trans2, trans_feat2 = self.feat(x2)
        # x = torch.cat((x1, x2), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out1 = F.relu(self.fc3(x))
        out1 = F.sigmoid(self.fc4(out1))
        # out2 = F.relu(self.fc5(x))
        # out2 = self.fc6(out2)
        return out1
