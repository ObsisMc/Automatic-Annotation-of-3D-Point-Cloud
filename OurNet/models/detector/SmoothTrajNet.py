import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import common_utils.cfgs as Config
from OurNet.models.backbone3D.vfe.pillar_vfe import PillarVFE
from OurNet.models.backbone2D.pointpillar_scatter import PointPillarScatter


class SmoothTrajNet(nn.Module):
    """
    use data in world coordinate
    """

    def __init__(self, device, N=10):
        super().__init__()
        self.device = device
        self.N = N
        self.pillar_cfg = Config.load_train_pillar_cfg()
        self.pillar_vfe = Config.load_pillar_vfe()
        self.grid_size = self.pillar_vfe["GRID_SIZE"].astype(np.float)
        self.voxel_size = self.pillar_vfe["VOXEL_SIZE"]
        self.pfe = PillarVFE(self.pillar_cfg, self.pillar_vfe["NUM_POINT_FEATURES"], self.voxel_size,
                             device).to(device)
        self.psct = PointPillarScatter(self.pillar_cfg["NUM_FILTERS"][-1], self.grid_size)
        self.unet1d = Unet1D(N=N).to(device)
        self.bev_fe = BevFeature()

        self.out_channels = 512
        self.filter = [512, 256]
        self.conv1 = nn.Conv1d(self.filter[0], self.filter[1], kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(self.filter[1])
        self.linear1 = nn.Linear(self.filter[1] * N, 1280)
        self.linear2 = nn.Linear(1280, 512)
        self.linear3 = nn.Linear(512, N * 3)

    def forward(self, point_dicts, poses):
        """
        points_dicts: [{key:(B,C,) ,...},...]
        """
        # get bev
        batch = point_dicts[0]["voxels"].shape[0]
        for point_dict in point_dicts:
            assert point_dict.get("point_cloud_range") is not None
            # vrange = point_dict["point_cloud_range"]
            # grid_size = (vrange[:, 3:6] - vrange[:, 0:3]) / self.voxel_size.astype(np.float)
            # assert torch.all(grid_size.type(torch.int) == torch.tensor(self.grid_size))

            self.psct(self.pfe(point_dict, point_dict["point_cloud_range"]))

        # extract features of poses
        poses_f = self.unet1d(poses.float()).transpose(2, 1)  # (B,C=256,N=10)

        # extract features of bevs
        # only cat not stack
        bev_f = point_dicts[0]["spatial_features"].to(self.device)
        for i in range(1, self.N):
            bev_f = torch.cat([bev_f, point_dicts[i]["spatial_features"]], dim=1)
        # (B,N*C,H,W) -> (B,C,N)
        bev_f = self.bev_fe(bev_f)  # (B,C=256,N=10)

        # cat features and decoding
        total_f = torch.cat([bev_f, poses_f], dim=1)  # (B,C=512,N=10)
        total_f = F.leaky_relu(self.bn1(self.conv1(total_f)))
        total_f = total_f.view(total_f.shape[0], -1)
        total_f = F.relu(self.linear1(total_f))
        total_f = F.leaky_relu(self.linear2(total_f))
        total_f = self.linear3(total_f)

        return total_f


class BevFeature(nn.Module):
    def __init__(self, input_channels=640):
        super().__init__()
        self.num_filter = [input_channels, 256, 64]
        self.out_filter_half = 128
        self.maxpool_size = [2]
        # (B,640,70,70) -> (B, 256, 68, 68) -> (B, 256, 34, 34) -> (B, 64, 32, 32) -> (B, 64, 16, 16)
        self.conv1 = nn.Conv2d(input_channels, self.num_filter[1], 3)
        self.conv2 = nn.Conv2d(self.num_filter[1], self.num_filter[2], 3)
        self.bn1 = nn.BatchNorm2d(self.num_filter[1])
        self.bn2 = nn.BatchNorm2d(self.num_filter[2])
        self.maxpool1 = nn.MaxPool2d(self.maxpool_size[0])

        # (B,64, 16, 16) -> (B, 128, 32, 32)
        # (B,64, 32, 32) -> (B, 128, 32, 32)
        self.deconv1 = nn.ConvTranspose2d(64, self.out_filter_half, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, self.out_filter_half, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.out_filter_half)
        self.bn4 = nn.BatchNorm2d(self.out_filter_half)

        # -> (B, 256, 32, 32) -> (B, 256, 10)
        self.linear = nn.Linear(32 * 32, 10)

    def forward(self, x):
        """
        (B,C,H,W)
        """
        x1 = F.relu(self.bn1(self.conv1(x)))
        max1 = self.maxpool1(x1)

        x2 = F.relu(self.bn2(self.conv2(max1)))
        max2 = self.maxpool1(x2)

        x3 = F.relu(self.bn3(self.deconv1(max2)))
        x4 = F.relu(self.bn4(self.conv3(x2)))

        # try:
        x4 = torch.cat([x3, x4], dim=1).view(x4.shape[0], x4.shape[1]*2, -1)
        # except Exception as e:
        #     print("Error",x3.shape, x4.shape)
        #     raise e
        x4 = self.linear(x4)

        return x4


class Unet1D(nn.Module):
    """
    it can only support N=10
    """

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
        self.conv2 = nn.Conv1d(self.out_channels[0], 256, 1)
        self.bn2 = nn.BatchNorm1d(256)

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
