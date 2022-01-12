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
        self.mu, self.std = self.calcNorm()
        print("Finish dataset!\nmu:{}\nstd:{}".format(self.mu, self.std))

    def __getitem__(self, item):
        # label格式： dx, dy, dz, d\theta, confidence
        with open(self.labels_path + self.data_labels[item], "r") as f:
            lb = f.readline().rstrip("\n").split(" ")  # 注意转换数据的时候是否去掉了换行符
            frame = lb[0]
            label = [float(lb[i]) for i in range(1, len(lb))]
            # # 标准化
            # label = (np.array(label) - self.mu) / self.std
        vp = self.velodynes_path + self.data_velodynes[item]
        points_name = os.listdir(vp)  # 注意不要依赖于系统排序！！
        points_name = sorted(points_name, key=lambda x: int(x.strip("point").rstrip(".npy")), reverse=False)  # 按idx升序
        points1 = np.load(os.path.join(vp, points_name[0]))
        points2 = np.load(os.path.join(vp, points_name[1]))

        return [points1, points2], label, frame

    def calcNorm(self):
        tmp1 = [[] for _ in range(4)]
        tmp2 = [[]]
        for texti in range(len(self.data_labels)):
            with open(self.labels_path + self.data_labels[texti], "r") as f:
                lb = f.readline().rstrip("\n").split(" ")
                label = [float(lb[i]) for i in range(1, len(lb))]
                # 没有车时需要舍弃label
                if int(label[-1]) == 1:
                    for i in range(len(label) - 1):
                        tmp1[i].append(label[i])
                tmp2[0].append(label[-1])
        tmp1 = np.array(tmp1)
        tmp2 = np.array(tmp2)
        return np.r_[np.mean(tmp1, axis=1), np.mean(tmp2, axis=1)], np.r_[np.std(tmp1, axis=1), np.std(tmp2, axis=1)]

    def __len__(self):
        return self.data_num


def test1():
    gap = 3
    ln = 154
    checkl = []
    for i in range(1, gap + 1):
        offset = ((ln - i + 1) + ln - 1) * (i - 1) // 2
        for j in range(0, ln - gap):
            input, label, frame = dataset.__getitem__(offset + j)
            if int(frame) == 3:
                checkl.append(input)
            try:
                if label[-1] != 0 and label[-1] != 1:
                    print("wrong label value")
                if len(label) == 0:
                    print("empty label")

                gtpoint = np.load(cprp.format(j % 153))
                if not np.all(gtpoint == input[0]):
                    info = "unknown error"
                    if np.all(gtpoint == input[1]):
                        info = "reverse point"
                    print("wrong point {} ({})".format(j + offset, info))

            except:
                print("wrong in {}!!!!!!!".format(i))
    print(checkl)
    point = checkl[0][1]
    for j in range(1, len(checkl)):
        if not np.all(point == checkl[j][1]):
            print("different error points for frame {}".format(3))


cprp = "../Data/Mydataset/0000/groundtruth/Van_0/point{}.npy"
if __name__ == "__main__":
    dataset = MyDataSet()
    # input, label, frame = dataset.__getitem__(2)
    # print(label)
    # print(frame)

    # for i in range(100):
    #     input, label = dataset.__getitem__(i)
    #     print(label)
    # testpoints = np.load("../Data/Mydataset/training/velodyne/{:04}/point0.npy".format(1))
    # input, label = dataset.__getitem__(0)
    # print(np.all(input[0] == testpoints))
