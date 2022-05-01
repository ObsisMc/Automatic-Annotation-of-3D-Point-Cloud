import numpy as np
import common_utils.cfgs as Config
import utils


class Sampler():
    def __init__(self, pointsn=800):
        self.pointsn = pointsn

    def before_sample(self, points: np.ndarray, n):
        """
        points: (N,3)
        """
        if points.shape[0] <= n:
            padding = [(0, 0, 0)] * (n - points.shape[0])
            return np.r_[points, np.array(padding)]
        return None

    def random_sample(self, points, n=800):
        sample = self.before_sample(points, n)
        if sample is not None:
            return sample
        randomlist = np.random.choice([i for i in range(points.shape[0])], n)
        return points[randomlist]

    def fps(self, points: np.ndarray, npoint=800):
        """
        Input:
            xyz: pointcloud data, [N, C]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        points = points[:, :3]
        centroids = self.before_sample(points, npoint)
        if centroids is not None:
            return centroids

        N, C = points.shape
        centroids = np.zeros(npoint)
        distance = np.ones(N) * 1e10
        farthest = 0
        for i in range(npoint):
            # 更新第i个最远点
            centroids[i] = farthest
            # 取出这个最远点的xyz坐标
            centroid = points[farthest].reshape(-1, 3)
            # 计算点集中的所有点到这个最远点的欧式距离
            dist = np.sum((points - centroid) ** 2, axis=1)
            # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
            mask = dist < distance
            distance[mask] = dist[mask]
            # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
            farthest = np.argmax(distance)
        return points[centroids.astype(np.int)]

    def paddingTraj(self, poses: np.ndarray, points_dicts, acutal_size=10, max_traj_n=10):
        """
        @params:
        points_dict a list of dict
        """

        def paddingPoses(poses):
            if max_traj_n > acutal_size:
                padding = np.zeros((max_traj_n - acutal_size, 3))
                poses = np.r_[poses, padding]
            return poses

        def paddingPoints(points_dicts):
            if max_traj_n - acutal_size > 0:
                for i in range(acutal_size, max_traj_n):
                    points_dicts[i]["points"] = np.array([[0, 0, 0]])  # 3d
            return points_dicts

        return paddingPoses(poses), paddingPoints(points_dicts)
