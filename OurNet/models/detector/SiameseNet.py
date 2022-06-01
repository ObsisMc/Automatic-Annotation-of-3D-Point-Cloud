import copy

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
    def __init__(self, k=3):
        super().__init__()
        self.backbone = self.getBackone(k=k)

    def getBackone(self, k=3, mlp=[64, 128, 512, 1024]):
        seq = nn.Sequential()
        input_d = k
        for i, output_d in enumerate(mlp):
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
        x1_feat = copy.deepcopy(x1)
        x2_feat = copy.deepcopy(x2)
        x1_feat = self.pointfeat(x1_feat.permute(0, 2, 1))
        x2_feat = self.pointfeat(x2_feat.permute(0, 2, 1))

        return x1, x2


class PointNetSetAbstraction(nn.Module):
    '''
    如：npoint=128,radius=0.4,nsample=64,in_channle=128+3,mlp=[128,128,256],group_all=False
    128=npoint:points sampled in farthest point sampling
    0.4=radius:search radius in local region
    64=nsample:how many points inn each local region
    [128,128,256]=output size for MLP on each point
    '''

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        # nn.ModuleList，它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。
        # 你可以把任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，
        # 方法和 Python 自带的 list 一样，无非是 extend，append 等操作。
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]  #(不太理解为啥维度变了，前面都是B,N,C)???????
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)  # 将tensor的维度换位#（B,N,C）
        if points is not None:
            points = points.permute(0, 2, 1)  # （B,N,D）

        if self.group_all:  # 形成局部的group
            new_xyz, new_points = sample_group_all(xyz=xyz, points=points)
        else:
            new_xyz, new_points = sample_group(radius=self.radius, fps_point_n=self.npoint,
                                               ball_point_n=self.nsample, xyz=xyz, points=points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]

        # 以下是pointnet操作：
        # 对局部group中每一个点做MLP操作:
        # 利用1*1的2d卷积相当于把每个group当成一个通道，共npoint个通道
        # 对[C+D，nsample]的维度上做逐像素的卷积，结果相当于对单个c+D维度做1d的卷积
        for i, conv in enumerate(self.mlp_convs):
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # print(new_points.shape)
        # 最后进行局部的最大池化，得到局部的全局特征
        # 对每个group做一个max pooling得到局部的全局特征,得到的new_points:[B,3+D,npoint]
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points




def batch_fps(points: torch.Tensor, sample_n) -> np.ndarray:
    """
    Input:
        xyz: pointcloud data, [N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    device = points.device
    batch, num, channel = points.shape
    fps_n = sample_n

    fps_centroids = np.zeros((batch, fps_n))
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
        fps_centroids[b, :] = centroids.astype(np.int)
    return fps_centroids


def index_points(points: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S] or [B,Nb,S]
        Return:
            new_points:, indexed points data, [B, S, C] or [B,Nb,S,C]
    """

    device = points.device
    batch = idx.shape[0]
    _, n_points, _ = points.shape

    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)

    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1

    batch_matrix = torch.arange(batch, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    sample_points = points[batch_matrix, idx, :]
    return sample_points


def square_distance(src, dst):
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


def neighbour_ball(sample_n, radius: float, all_points: torch.Tensor, centroid_points):
    """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, 3]
            new_xyz: query points, [B, S, 3]
        Return:
            group_idx: grouped points index, [B, S, nsample]
    """
    device = all_points.device
    batch, n_all, n_channel = all_points.shape
    _, n_centroid, _ = centroid_points.shape

    group_idx = torch.arange(n_all, dtype=torch.long).to(device).view(1, 1, n_all).repeat([batch, n_centroid, 1])
    dist = square_distance(centroid_points, all_points)
    group_idx[dist > radius ** 2] = n_all
    group_idx = group_idx.sort(dim=-1)[0][:, :, :sample_n]

    group_first = group_idx[:, :, 0].view(batch, n_centroid, 1).repeat(1, 1, sample_n)
    mask = group_idx == n_all
    group_idx[mask] = group_first[mask]  # can there be group_idx = group_first
    return group_idx


def sample_group(radius, fps_point_n, ball_point_n, xyz, points):
    batch, all_n, channel_n = xyz.shape
    fps_idx = batch_fps(xyz, fps_point_n)
    fps_points = index_points(xyz, fps_idx)

    group_idx = neighbour_ball(sample_n=ball_point_n, radius=radius,
                               all_points=xyz, centroid_points=fps_points)
    group_xyz = index_points(xyz, group_idx)
    group_xyz_norm = group_xyz - fps_points.view(batch, fps_point_n, 1, channel_n)

    if points is None:
        new_feat_points = group_xyz_norm
    else:
        group_feat_points = index_points(points, group_idx)
        new_feat_points = torch.cat([group_feat_points, group_xyz_norm], dim=-1)

    return fps_points, new_feat_points


def sample_group_all(xyz: torch.Tensor, points: torch.Tensor):
    batch, all_n, channel_n = points.shape
    device = points.device
    centroid = torch.zeros([batch, 1, channel_n]).to(device)
    group_xyz = xyz.view(batch, 1, all_n, channel_n)
    group_xyz_norm = group_xyz - centroid.view(batch, 1, -1, channel_n)

    if points is None:
        new_points = group_xyz_norm
    else:
        new_points = torch.cat([points.view(batch, 1, all_n, -1), group_xyz_norm], dim=-1)
    return centroid, new_points
