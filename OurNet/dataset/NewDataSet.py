import numpy as np
from torch.utils.data import Dataset
from dataset_utils import DatasetCreator as datacreator, Augmentor as augmentor, Sampler


class NewDataSet(Dataset):
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
        self.data_list = self.create_data(datapath)
        self.sampler = Sampler.Sampler()

    def __getitem__(self, item):
        """
        return: (2,N,3)
        """
        data_root = self.data_list[item]
        source, raw_target = self.sampler.fps(np.load(data_root[0])), self.sampler.fps(
            np.load(data_root[1]))
        target, label = augmentor.guassianArgu(raw_target)  # label: dx, dy, dz, angle, confidence
        return [source, target], label

    def __len__(self):
        return len(self.data_list)

    def create_data(self, datapath):
        return datacreator.gapTwoInTradj(datapath=datapath)
