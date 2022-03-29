import numpy as np
import utils


def guassianArgu(points: np.ndarray):
    x_error = np.random.normal(loc=0, scale=1, size=None)
    y_error = np.random.normal(loc=0, scale=1, size=None)
    # z_error = np.random.normal(loc=0, scale=1, size=None)
    z_error = 0
    angle = np.random.normal(loc=0, scale=0.3, size=None)
    confidence = 1.0
    points = utils.rotate_points_along_z(points + np.array([x_error, y_error, z_error]), angle)
    return points, np.array([-x_error, -y_error, -z_error, -angle, confidence])
