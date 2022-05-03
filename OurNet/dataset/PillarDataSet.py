import DataSetTemplate
import numpy as np
import common_utils.cfgs as Config
from dataset_utils.processor.data_processor import DataProcessor


class SmPillarDataSet(DataSetTemplate.DataSetTemplate):
    """
    1. need to use data in its own coordinate.
    2. data should be always with points (if the data has no point, we will raise an exception)
    """

    def __init__(self, datapath):
        super().__init__(datapath)
        self.data_list = self.datasetcreator.gapTwoInTradj()
        self.data_processor = DataProcessor(Config.load_pillar_vfe())
        self.source_dict, self.target_dict = Config.load_pillar_data_template(2)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        source_root, target_root = self.data_list[index]

        # no need to sample, since transform will do it
        self.source_dict["points"] = np.load(source_root)
        self.target_dict["points"] = np.load(target_root)
        self.data_processor.transform_points_to_voxels(self.source_dict)
        self.data_processor.transform_points_to_voxels(self.target_dict)

        if  self.source_dict["voxel_num_points"].shape[0] == 0 or self.target_dict["voxel_num_points"].shape[0] == 0:
            raise AssertionError("no voxels in index %d" % (index))

        return self.source_dict, self.target_dict


class SmPillarSizeDataSet(DataSetTemplate.DataSetTemplate):
    def __init__(self, dataPath):
        super().__init__(dataPath)
        self.point_list, self.label_list = self.datasetcreator.severalFrameInTraj()
        self.data_processor = DataProcessor(Config.load_pillar_vfe())
        self.source_dict = Config.load_pillar_data_template(1)[0]

    def __len__(self):
        return len(self.point_list)

    def __getitem__(self, index):
        point_root = self.point_list[index]
        # merge all frames into ndarray points
        points = np.zeros((0, 3))
        # i = 0
        for root in point_root:
            # i += 1
            read_point = np.load(root)
            # add a column of i to the right of read_point
            # read_point = np.concatenate((read_point, np.full((read_point.shape[0], 1), i)), axis=1)
            points = np.concatenate((points, read_point))
        self.source_dict["points"] = points
        self.data_processor.transform_points_to_voxels(self.source_dict)
        # assert self.source_dict["voxel_num_points"].shape[0] > 0
        return self.source_dict, self.label_list[index]


def test1():
    spd = SmPillarDataSet(Config.load_train_common()["dataset_path"])
    source, target = spd[0]
    print(source)


if __name__ == "__main__":
    test1()
