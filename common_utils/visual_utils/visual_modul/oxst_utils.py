import numpy as np
from pyproj import Proj  # wgs84->utm transformation


class OxstProjector:
    def __init__(self, zone=32):
        """
         @params: zone, latitude and longitude to zone https://www.latlong.net/lat-long-utm.html
        """
        self.__wgs84_proj = Proj(proj='utm', zone=zone, ellps='WGS84', preserve_units=False)

    def oxst_to_coord(self, latitude, longitude):
        x, y = self.__wgs84_proj(longitude, latitude)
        return x, y


if __name__ == "__main__":
    lat, long = 49.011212804408, 8.4228850417969
    proj = OxstProjector()
    print(proj.oxst_to_coord(lat, long))
    pass
