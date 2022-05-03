import DataSetTemplate
import numpy as np
import common_utils.cfgs as Config
import cv2
import random


class SimpleSizeDataSet(DataSetTemplate.DataSetTemplate):
    def __init__(self, dataPath):
        super().__init__(dataPath)
        self.point_list, self.label_list = self.datasetcreator.severalFrameInTraj()

    def __len__(self):
        return len(self.point_list)

    def __getitem__(self, index):
        bev = np.zeros((128, 128))
        point_root = self.point_list[index]
        j = 0
        for cloud_file in point_root:
            j += 1
            gauss_x = random.gauss(0, 0.04)
            gauss_y = random.gauss(0, 0.02)
            cloud = np.load(cloud_file)
            for i in range(cloud.shape[0]):
                x = int((cloud[i][0] + gauss_x) / 0.05) + 64
                y = int((cloud[i][1] + gauss_y) / 0.05) + 64
                if 0 <= x < 128 and 0 <= y < 128:
                    # bev[x][y] += cloud[i][2]
                    bev[x][y] += 1
        bev = bev.astype(np.uint8)
        bev = cv2.equalizeHist(bev)
        bev = (bev - np.mean(bev)) / np.std(bev)
        bev = bev.astype(np.float32)
        return bev, self.label_list[index] / 1.3
