import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SimplePointNet(nn.Module):
    def __init__(self, k=3):
        super(SimplePointNet, self).__init__()
        self.backbone = self.getBackone(k=k)

    def getBackone(self, k=3, mlp=[64, 128, 512, 1024]):
        seq = nn.Sequential()
        input_d = k
        for i, output_d in enumerate(mlp):
            seq.add_module("conv%d" % (i + 1), nn.Conv2d(input_d, output_d, (1, 1)))
            seq.add_module("bn%d" % (i + 1), nn.BatchNorm2d(output_d))
            seq.add_module("relu%d" % (i + 1), nn.LeakyReLU())
            input_d = output_d
        return seq

    def forward(self, x):
        """
        x: (B,3,N,C)
        """
        x = self.backbone(x)
        x = torch.max(x, 2, keepdim=False)[0]
        x = x.view(-1, 1024 * x.shape[2])
        return x


class Siamese2c(nn.Module):
    def __init__(self, k=5):
        super(Siamese2c, self).__init__()
        self.pointfeat = SimplePointNet()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, k)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x1, x2):
        """
        x1 and x2: (B,N,3)
        """
        x = torch.cat((x1.unsqueeze(3), x2.unsqueeze(3)), 3).permute(0, 2, 1, 3)
        x = self.pointfeat(x)
        x = F.leaky_relu(self.dropout(self.fc1(x)))
        x = F.leaky_relu(self.dropout(self.fc2(x)))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        return x


class SiameseMultiDecoder(nn.Module):
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2048, 512)
            self.fc2 = nn.Linear(512, 64)
            self.fc3 = nn.Linear(64, 1)

            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(512)
            self.bn3 = nn.BatchNorm1d(64)

        def forward(self, x):
            x = F.leaky_relu(self.bn2(self.fc1(x)))
            x = F.leaky_relu(self.bn3(self.fc2(x)))
            x = self.fc3(x)
            return x

    def __init__(self, k=5):
        super().__init__()
        self.pointfeat = SimplePointNet()
        self.k = k
        self.decoders = [self.Decoder() for _ in range(k)]  # x,y,z,angel,confidence

    def forward(self, x1, x2):
        """
        x1 and x2: (B,N,3)
        """
        x = torch.cat((x1.unsqueeze(3), x2.unsqueeze(3)), 3).permute(0, 2, 1, 3)
        x = self.pointfeat(x)

        y = [self.decoders[i].to(x.device)(x) for i in range(self.k)]
        return y


class PointNet1D(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.backbone = self.getBackone(k=k)

    def getBackone(self, k=3, mlp=[64, 128, 512, 1024]):
        seq = nn.Sequential()
        input_d = k
        for i, output_d in enumerate(mlp):
            print(i, output_d)
            seq.add_module("conv%d" % (i + 1), nn.Conv1d(input_d, output_d, 1))
            seq.add_module("bn%d" % (i + 1), nn.BatchNorm1d(output_d))
            seq.add_module("relu%d" % (i + 1), nn.LeakyReLU())
            input_d = output_d
        return seq

    def forward(self, x):
        x = self.backbone(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024 * x.shape[2])
        return x


class SiamesePlus(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.pointfeat = PointNet1D()

    def forward(self, x1, x2):
        x1 = self.pointfeat(x1.permute(0, 2, 1))
        x2 = self.pointfeat(x2.permute(0, 2, 1))
        return x1, x2

    def batch_fps(self, points: torch.Tensor):
        """
        Input:
            xyz: pointcloud data, [N, C]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """

        device = points.device
        batch, num, channel = points.shape
        fps_n = num // 2

        fps_points = np.zeros((batch, fps_n))
        for b in batch:
            centroids = np.zeros(fps_n)
            distance = np.ones(fps_n) * 1e10
            farthest = 0
            for i in range(fps_n):
                # 更新第i个最远点
                centroids[i] = farthest
                # 取出这个最远点的xyz坐标
                centroid = points[b, farthest].reshape(-1, 3)
                # 计算点集中的所有点到这个最远点的欧式距离
                dist = np.sum((points[b, :] - centroid) ** 2, axis=1)
                # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
                mask = dist < distance
                distance[mask] = dist[mask]
                # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
                farthest = np.argmax(distance)
            fps_points[b, :] = points[centroids.astype(np.int)]
        return fps_points

    def square_distance(self, src, dst):
        # 这里计算的是原点（src）集合中N个点到目标点（dst）集合中M点的距离（平方距离，没有开根号），以Batch为单位，输出B×N×M的tensor。
        """
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape  # 单下划线表示不关心
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # permute为转置,[B, N, M]
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        # [B, N, M] + [B, N, 1]，dist的每一列都加上后面的列值
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)  # [B, N, M] + [B, 1, M],dist的每一行都加上后面的行值
        return dist

    def neighbour_ball(self):

        pass
