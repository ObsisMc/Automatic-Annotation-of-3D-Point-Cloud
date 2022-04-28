import DataSetTemplate
import numpy as np
import common_utils.cfgs as Config
from dataset_utils.processor.data_processor import DataProcessor


class SmPillarDataSet(DataSetTemplate.DataSetTemplate):
    """

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

        assert self.source_dict["voxel_num_points"].shape[0] > 0 and self.target_dict["voxel_num_points"].shape[0] > 0

        return self.source_dict, self.target_dict


def test1():
    spd = SmPillarDataSet(Config.load_train_common()["dataset_path"])
    source, target = spd[0]
    print(source)


if __name__ == "__main__":
    test1()
