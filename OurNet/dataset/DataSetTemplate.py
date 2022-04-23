import torch
from torch.utils.data import Dataset
import common_utils.cfgs as Config
import OurNet.dataset.dataset_utils as dataset_utils
from dataset_utils import Augmentor, DatasetCreator, Sampler


class DataSetTemplate(Dataset):
    def __init__(self, datapath):
        super().__init__()
        self.augmentor = Augmentor.Augmentor()
        self.datasetcreator = DatasetCreator.DataSetCreator(datapath)
        self.sampler = Sampler.Sampler()

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, item):
        return NotImplementedError
