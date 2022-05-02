import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import common_utils.cfgs as Config
from OurNet.models.backbone3D.vfe.pillar_vfe import PillarVFE
from OurNet.models.backbone2D.pointpillar_scatter import PointPillarScatter


class SmoothTrajNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.pillar_cfg = Config.load_train_pillar_cfg()
        self.pillar_vfe = Config.load_pillar_vfe()
        self.grid_size = self.pillar_vfe["GRID_SIZE"].astype(np.float)
        self.voxel_size = self.pillar_vfe["VOXEL_SIZE"]
        self.pfe = PillarVFE(self.pillar_cfg, self.pillar_vfe["NUM_POINT_FEATURES"], self.voxel_size,
                             device).to(device)
        self.psct = PointPillarScatter(self.pillar_cfg["NUM_FILTERS"][-1], self.grid_size)
        self.unet1d = Unet1D()

    def forward(self, point_dicts, poses):
        """
        points_dicts: [{key:(B,C,) ,...},...]
        """
        batch = point_dicts[0]["voxels"].shape[0]
        for point_dict in point_dicts:
            assert point_dict.get("point_cloud_range") is not None
            # vrange = point_dict["point_cloud_range"]
            # grid_size = (vrange[:, 3:6] - vrange[:, 0:3]) / self.voxel_size.astype(np.float)
            # assert torch.all(grid_size.type(torch.int) == torch.tensor(self.grid_size))

            self.psct(self.pfe(point_dict, point_dict["point_cloud_range"]))

        print("bev:", point_dicts[0]["spatial_features"].shape)
        return point_dicts


class Unet1D(nn.Module):
    def __init__(self, N=10, in_channels=3):
        super().__init__()
        self.out_channels = [64, 128, 256, 512]
        self.doubleConv1 = DoubleConv1D(in_channels, self.out_channels[0])  # -> (8, 64)
        self.doubleConv2 = DoubleConv1D(self.out_channels[0], self.out_channels[1])  # (6, 128)
        self.doubleConv3 = DoubleConv1D(self.out_channels[1], self.out_channels[2])  # (4, 256)
        # self.doubleConv4 = DoubleConv1D(self.out_channels[2], self.out_channels[3])  # N -> 2

        # self.deConv1 = DeConv1D(self.out_channels[3], self.out_channels[2], padding=1)  # N -> 4
        self.deConv2 = DeConv1D(self.out_channels[2], self.out_channels[1])  # (6,128)
        self.deConv3 = DeConv1D(self.out_channels[1], self.out_channels[0])  # (10,64)
        self.deConv4 = DeConv1D(self.out_channels[0], self.out_channels[3])

        self.conv1 = nn.Conv1d(self.out_channels[2], self.out_channels[1], 1)  # (6,256) -> (6,128)
        self.bn1 = nn.BatchNorm1d(128)
        # self.conv2 = nn.Conv1d(self.out_channels[1], self.out_channels[0], 1)
        # self.bn2 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(self.out_channels[0], self.out_channels[3], 1)
        self.bn2 = nn.BatchNorm1d(self.out_channels[3])

    def forward(self, x):
        """
        (B,N,3) -> (B, N, 512)
        """
        x1 = self.doubleConv1(x)
        x2 = self.doubleConv2(x1)
        x3 = self.doubleConv3(x2)

        dx2 = self.deConv2(x3)
        x_mid = torch.cat([x2, dx2], dim=2)
        x_mid = F.leaky_relu(self.bn1(self.conv1(x_mid.transpose(2, 1)))).transpose(2, 1)

        dx1 = self.deConv3(x_mid)
        # x_mid = torch.cat([dx1, x1], dim=2)
        # x_mid = F.leaky_relu(self.bn2(self.conv2(x_mid.transpose(2, 1)))).transpose(2, 1)

        x_res = F.leaky_relu(self.bn2(self.conv2(dx1.transpose(2, 1)))).transpose(2, 1)
        return x_res


class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        mid_channels = (out_channels - in_channels) // 2
        self.conv1d1 = nn.Conv1d(in_channels, mid_channels, kernel_size)
        self.conv1d2 = nn.Conv1d(mid_channels, out_channels, kernel_size)

        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        (B,N,3)
        """
        x = x.transpose(2, 1)
        x = self.conv1d1(x)
        x = F.leaky_relu(self.bn1(x))
        x = F.leaky_relu(self.bn2(self.conv1d2(x)))
        x = x.transpose(2, 1)
        return x


class DeConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=2, stride=2):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        (B,N,3)
        """
        x = F.leaky_relu(self.bn(self.deconv1(x.transpose(2, 1))))
        return x.transpose(2, 1)


if __name__ == "__main__":
    l = []
    for i in range(10):
        l.append([i for _ in range(3)])
    l = [l]
    test = torch.tensor(l, dtype=torch.float32)  # need float32

    # conv1 = nn.Conv1d(3, 10, kernel_size=2)
    # res = conv1(test.transpose(2, 1))
    unet = Unet1D()
    res = unet(test)
    print(res)
