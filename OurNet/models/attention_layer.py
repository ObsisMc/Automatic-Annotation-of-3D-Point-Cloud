import torch
import torch.nn as nn
import torch.nn.functional as F


class SEnet(nn.Module):
    def __init__(self, in_channel, scale):
        super(SEnet, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(in_channel, in_channel // scale),
            nn.ReLU(),
            nn.Linear(in_channel // scale, in_channel)
        )
        self.activate = nn.Sigmoid()

    def forward(self, x):
        """
        @params: x (B,C,H,W)

        @return: out (B,C)
        """
        x = self.linear(self.avgPool(x))
        out = self.activate(x)
        return out

