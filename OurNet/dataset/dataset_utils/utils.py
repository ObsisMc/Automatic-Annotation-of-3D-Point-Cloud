import numpy as np


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    ones = np.ones_like(angle, dtype=np.float32)
    zeros = np.zeros_like(angle, dtype=np.float32)
    rot_matrix = np.array(
        [[cosa, sina, zeros],
         [-sina, cosa, zeros],
         [zeros, zeros, ones]]
    )
    points_rot = points @ rot_matrix
    return points_rot
