import numpy as np
import utils


class Augmentor():
    def guassianAug(self, points: np.ndarray):
        x_error = np.random.normal(loc=0, scale=1, size=None)
        y_error = np.random.normal(loc=0, scale=1, size=None)
        # z_error = np.random.normal(loc=0, scale=1, size=None)
        z_error = 0
        angle = np.random.normal(loc=0, scale=0.3, size=None)
        confidence = 1.0
        points = utils.rotate_points_along_z(points + np.array([x_error, y_error, z_error]), angle)
        return points, np.array([-x_error, -y_error, -z_error, -angle, confidence])

    def guassianTrajAug(self, poses, points_dicts, max_size: int, actual_size: int):
        """
        used by SmoothTrajDataSet
        IMPORTANT!!!! error shouldn't be too large
        """

        x_error = np.random.normal(loc=0, scale=0.1, size=max_size).reshape(-1, 1)
        y_error = np.random.normal(loc=0, scale=0.1, size=max_size).reshape(-1, 1)
        angle = np.random.normal(loc=0, scale=0.01, size=max_size).reshape(-1, 1)
        # x_error = np.random.normal(loc=0, scale=0.1, size=max_size).reshape(-1, 1)
        # y_error = np.random.normal(loc=0, scale=0.1, size=max_size).reshape(-1, 1)
        # angle = np.array([0.03 for _ in range(max_size)]).reshape(-1, 1)

        error = np.c_[x_error, y_error, angle]

        def augPoses(poses):
            for i in range(1, max_size):
                poses[i] += error[i] - error[i - 1]
            return poses

        def augPoints(points_dicts):
            for i in range(actual_size):
                points = points_dicts[i]["points"]
                points = points + np.array([x_error[i, 0], y_error[i, 0], 0]).reshape(1, -1)
                points_dicts[i]["points"] = utils.rotate_points_along_z(points, angle[i, 0])
            for i in range(actual_size, max_size):
                points_dicts[i]["points"] = points_dicts[actual_size - 1]["points"]
            return points_dicts

        return augPoses(poses), augPoints(points_dicts), -error
