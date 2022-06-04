import numpy as np
import utils


class Augmentor:
    def guassianAug(self, points: np.ndarray, conf=1):
        confidence = np.random.randint(2) if conf is None else conf
        x_error = y_error = angel = z_error = 0
        if confidence == 1:
            x_error = np.random.normal(loc=0, scale=0.16, size=None)
            y_error = np.random.normal(loc=0, scale=0.16, size=None)
            angel = np.random.normal(loc=0, scale=0.17, size=None)
        else:
            sd = np.random.randint(2)
            if sd:
                x_error = np.random.normal(loc=0, scale=1, size=None)
                y_error = np.random.normal(loc=0, scale=1, size=None)
                angel = np.random.normal(loc=0, scale=1, size=None)

                num = points.shape[0]
                sample_n = num // 100 if num > 100 else num // 10
                sample_idx = np.random.choice(np.arange(num), size=sample_n)
                points = points[sample_idx, :]

                padding = np.zeros((num - sample_n, 3))
                points = np.r_[points, padding].astype(np.double)
            else:
                points = np.zeros((800, 3), dtype=np.double)
        points = utils.rotate_points_along_z(points + np.array([x_error, y_error, z_error]), angel)
        return points, np.array([-x_error, -y_error, -z_error, -angel, confidence])

    def guassianTrajAug(self, poses, points_dicts, centers, max_size: int, actual_size: int):
        """
        used by SmoothTrajDataSet
        IMPORTANT!!!! error shouldn't be too large
        """

        x_error = np.random.normal(loc=0, scale=0.1, size=max_size).reshape(-1, 1)
        y_error = np.random.normal(loc=0, scale=0.1, size=max_size).reshape(-1, 1)
        angle = np.random.normal(loc=0, scale=0.1, size=max_size).reshape(-1, 1)

        error = np.c_[x_error, y_error, angle]

        def augPoses(poses):
            for i in range(1, max_size):
                poses[i] += error[i] - error[i - 1]
            return poses

        def augPoints(points_dicts, centers):
            for i in range(actual_size):
                points = points_dicts[i]["points"] - centers[i].reshape(1, -1)
                points = utils.rotate_points_along_z(points, angle[i, 0])
                points_dicts[i]["points"] = points + centers[i].reshape(1, -1) + \
                                            np.array([x_error[i, 0], y_error[i, 0], 0]).reshape(1, -1)
            for i in range(actual_size, max_size):
                points_dicts[i]["points"] = points_dicts[actual_size - 1]["points"]
            return points_dicts

        return augPoses(poses), augPoints(points_dicts, centers), -error
