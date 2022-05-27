import numpy as np
from pyproj import Proj  # wgs84->utm transformation


class OxstProjector:
    def __init__(self, zone=32):
        """
         @params: zone, latitude and longitude to zone https://www.latlong.net/lat-long-utm.html
        """
        self.__wgs84_proj = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units=False)
        self.yaw = None
        self.x, self.y = None, None
        self.base_position = None

    def oxst_to_coord(self, latitude, longitude):
        x, y = self.__wgs84_proj(longitude, latitude)
        return x, y

    def init_oxst(self, oxst_config):
        latitude, longitude, yaw = float(oxst_config[0]), float(oxst_config[1]), float(oxst_config[5])
        self.x, self.y = self.oxst_to_coord(latitude, longitude)
        self.yaw = yaw
        if self.base_position is None:
            self.base_position = np.array([self.x, self.y, 0])
        return [self.x, self.y, self.yaw]

    def lidar_to_pose(self, object_points, base_position=None):
        """
        lidar_box: [x, y, z, dx, dy, dz, heading]
        base_position: np.ndarray, (-1,), [base_x, base_y]
        """
        assert self.yaw and self.x and self.y
        object_points = self.rotate_yaw(object_points, self.yaw)  # pay attention, yaw doesn't need to be negative

        if base_position is not None:
            base_position = base_position.astype(np.float)
        else:
            base_position = self.base_position

        offset = (np.array([self.x, self.y, 0]) - base_position).reshape(1, -1)
        object_points += offset
        return object_points

    def oxst_rotation_y(self, rotation_y):
        return rotation_y + self.yaw

    def rotate_yaw(self, points, angle):
        """
        points: (N,3)
        """
        cosa = np.cos(angle)
        sina = np.sin(angle)
        ones = np.ones_like(angle, dtype=np.float32)
        zeros = np.zeros_like(angle, dtype=np.float32)
        # clockwise
        rot_matrix = np.array(
            [[cosa, sina, zeros],
             [-sina, cosa, zeros],
             [zeros, zeros, ones]]
        )
        points_rot = points @ rot_matrix
        return points_rot


if __name__ == "__main__":
    lat, long = 49.011212804408, 8.4228850417969
    proj = OxstProjector()
    print(proj.oxst_to_coord(lat, long))
    pass
