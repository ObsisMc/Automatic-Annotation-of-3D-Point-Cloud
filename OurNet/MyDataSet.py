from torch.utils.data import Dataset
import numpy as np


class MyDataSet(Dataset):
    def __init__(self, data_path):
        self.data = self.initdata(data_path)
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.data)

    def initdata(self, data_path):
        data = None
        return data
