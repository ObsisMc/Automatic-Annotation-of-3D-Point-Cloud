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
        self.data_num = len(self.data_labels)

    def __getitem__(self, item):
        # label格式： dx, dy, dz, d\theta, confidence
        with open(self.labels_path + self.data_labels[item], "r") as f:
            lb = f.readline().rstrip("\n").split(" ")  # 注意转换数据的时候是否去掉了换行符
            label = [float(lb[i]) for i in range(1, len(lb))]
        vp = self.velodynes_path + self.data_velodynes[item]
        points_name = os.listdir(vp)  # 注意不要依赖于系统排序！！
        points_name = sorted(points_name, key=lambda x: int(x.strip("point").rstrip(".npy")), reverse=False)  # 按idx升序
        points1 = np.load(os.path.join(vp, points_name[0]))
        points2 = np.load(os.path.join(vp, points_name[1]))

        return [points1, points2], label

    def __len__(self):
        return self.data_num


cprp = "../Data/Mydataset/0000/groundtruth/Van_0/point{}.npy"
if __name__ == "__main__":
    dataset = MyDataSet()
    # 只能测试gap为1的数据
    for i in range(len(dataset)):
        input, label = dataset.__getitem__(i)
        try:
            if label[-1] != 0 and label[-1] != 1:
                print("wrong label value")
            if len(label) == 0:
                print("empty label")

            gtpoint = np.load(cprp.format(i % 153))
            if not np.all(gtpoint == input[0]):
                info = "unknown error"
                if np.all(gtpoint == input[1]):
                    info = "reverse point"
                print("wrong point {} ({})".format(i, info))

        except:
            print("wrong in {}!!!!!!!".format(i))
    # for i in range(100):
    #     input, label = dataset.__getitem__(i)
    #     print(label)
    # testpoints = np.load("../Data/Mydataset/training/velodyne/{:04}/point0.npy".format(1))
    # input, label = dataset.__getitem__(0)
    # print(np.all(input[0] == testpoints))
