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

    def __init__(self, datapath="/home/zrh/Data/kitti/tracking/extracted_points_default"):
        super().__init__(datapath)
        self.data_list = self.datasetcreator.gapTwoInTradj(num_threshold=None)
        NewDataSet.statics_analyse(self.data_list)

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

    @staticmethod
    def statics_analyse(dataset: list):
        max_n = 0
        min_n = 100000
        sum_n = 0
        num = 0
        for i, data in enumerate(dataset):
            source, target = np.load(data[0]), np.load(data[1])

            tmp_num = source.shape[0]
            sum_n += tmp_num
            num += 1
            max_n = max(tmp_num, max_n)
            min_n = min(tmp_num, min_n)

            if i == len(dataset) - 1:
                tmp_num = target.shape[0]
                sum_n += tmp_num
                num += 1
                max_n = max(tmp_num, max_n)
                min_n = min(tmp_num, min_n)

        avg_n = sum_n / num
        print("num of point clouds: %d, max: %d, min: %d, avg: %f" % (num, max_n, min_n, avg_n))


if __name__ == "__main__":
    dataset = NewDataSet("/home/zrh/Data/kitti/tracking/extracted_points_keep")
    print(len(dataset))
    for i in range(len(dataset)):
        points, label = dataset[i]
        if i % 500 == 0:
            print("pass %d" % (i))
    print("finish")
