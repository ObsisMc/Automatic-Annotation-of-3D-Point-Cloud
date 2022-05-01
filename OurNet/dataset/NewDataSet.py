import numpy as np
from DataSetTemplate import DataSetTemplate


class NewDataSet(DataSetTemplate):
    """
    data_root
      |-> 0000 // scene
            |-> Car#1 // type and trajectoryId
                  |-> points
                    |-> 000000.npy // points cloud in frameId
                  |-> labels
                    |-> 000000.txt
    """

    def __init__(self, datapath):
        super().__init__(datapath)
        self.data_list = self.datasetcreator.gapTwoInTradj()

    def __getitem__(self, item):
        """
        return: (2,N,3)
        """
        data_root = self.data_list[item]
        source, raw_target = self.sampler.fps(np.load(data_root[0])), self.sampler.fps(
            np.load(data_root[1]))
        target, label = self.augmentor.guassianAug(raw_target)  # label: dx, dy, dz, angle, confidence
        return [source, target], label

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    dataset = NewDataSet("/home/zrh/Data/kitti/tracking/extracted_points_keep")
    print(len(dataset))
    for i in range(len(dataset)):
        points, label = dataset[i]
        if i % 500 == 0:
            print("pass %d" % (i))
    print("finish")
