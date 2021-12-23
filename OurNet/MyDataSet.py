import os

import numpy as np
from torch.utils.data import Dataset

error_dir = "../Data/Mydataset/{:04}/error"
gt_dir = "../Data/Mydataset/{:04}/groundtruth"
label_dir = "../Data/Mydataset/{:04}/label"


class MyDataSet(Dataset):
    def __init__(self, data_path="../Data/Mydataset/training/"):
        super(MyDataSet, self).__init__()
        self.velodynes_path = data_path + "velodyne/"
        self.labels_path = data_path + "label/"
        self.data_velodynes = os.listdir(self.velodynes_path)
        self.data_labels = os.listdir(self.labels_path)  # 获取label列表

    def __getitem__(self, item):
        # label格式： dx, dy, dz, d\theta, confidence
        with open(self.labels_path + self.data_labels[item], "r") as f:
            lb = f.readlines()  # 注意转换数据的时候是否去掉了换行符
            label = [int(lb[i]) for i in range(1, len(lb))]
            label.append(1)
        vp = self.velodynes_path + self.data_velodynes[item]
        points_name = os.listdir(vp)
        points1 = np.load(os.path.join(vp, points_name[0]))
        points2 = np.load(os.path.join(vp, points_name[1]))

        return [points1, points2], label

    def __len__(self):
        return self.data_num


if __name__ == "__main__":
    dataset = MyDataSet()
    for i in range(100):
        input, label = dataset.__getitem__(0)
        print(label)

