from functools import partial

import numpy as np
from skimage import transform

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, pillar_config):
        self.point_cloud_range = pillar_config["POINT_CLOUD_RANGE"]
        self.num_point_features = pillar_config["NUM_POINT_FEATURES"]
        self.mode = pillar_config["TRAIN"]
        self.grid_size = self.voxel_size = pillar_config["VOXEL_SIZE"]
        self.max_points_per_voxel = pillar_config["MAX_POINTS_PER_VOXEL"]
        self.max_number_of_voxels = pillar_config["MAX_NUMBER_OF_VOXELS"]

        self.voxel_generator = None

    def transform_points_to_voxels(self, data_dict, vsize_xyz=None, coors_range_xyz=None):
        variable = coors_range_xyz is not None or vsize_xyz is not None
        if self.voxel_generator is None or variable:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=vsize_xyz if vsize_xyz is not None else self.voxel_size,
                coors_range_xyz=coors_range_xyz if coors_range_xyz is not None else self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=self.max_points_per_voxel,
                max_num_voxels=self.max_number_of_voxels[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict
