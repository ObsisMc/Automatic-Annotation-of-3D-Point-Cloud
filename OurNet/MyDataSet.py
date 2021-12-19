import os

import numpy as np
from torch.utils.data import Dataset

error_dir = "../Data/Mydataset/{:04}/error"
gt_dir = "../Data/Mydataset/{:04}/groundtruth"
label_dir = "../Data/Mydataset/{:04}/label"


class MyDataSet(Dataset):
    def __init__(self, data_path):
        super(MyDataSet, self).__init__()
        self.data_path = data_path
        self.data_name = os.listdir(data_path)  # A list of names for data files
        self.data_num = len(self.data_name)  # The number of data files

    def __getitem__(self, item):
        data_name = self.data_name[item]
        data_path = os.path.join(self.data_path, data_name)
        data = np.load(data_path)
        return data

    def __len__(self):
        return self.data_num


if __name__ == "__main__":
    print(os.listdir(error_dir.format(0)))
