import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

import common_utils.cfgs as Config
from OurNet.models.backbone3D.vfe.pillar_vfe import PillarVFE
from OurNet.models.backbone2D.pointpillar_scatter import PointPillarScatter


class SmPillarNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.pillar_cfg = Config.load_train_pillar_cfg()
        self.pillar_vfe = Config.load_pillar_vfe()
        self.pfe = PillarVFE(self.pillar_cfg, self.pillar_vfe["NUM_POINT_FEATURES"],
                             self.pillar_vfe["VOXEL_SIZE"], device).to(device)  # extract pillar's features
        # map pillar to bev. IMPORTANT: grid_size need to be consistent with point_cloud_range
        self.psct = PointPillarScatter(self.pillar_cfg["NUM_FILTERS"][-1],
                                       self.pillar_vfe["GRID_SIZE"])
        pass

    def forward(self, source_dict, target_dict):
        # get features
        source_dict, target_dict = self.pfe(source_dict, self.pillar_vfe["POINT_CLOUD_RANGE"]), self.pfe(target_dict,
                                                                                                         self.pillar_vfe[
                                                                                                             "POINT_CLOUD_RANGE"])
        # map to bev
        source_dict, target_dict = self.psct(source_dict), self.psct(target_dict)
        sbev, tbev = source_dict["spatial_features"], target_dict["spatial_features"]  # (B,C,H,W)
        return sbev


class SmPillarSizeNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        pillar_cfg = Config.load_train_pillar_cfg()
        pillar_vfe = Config.load_pillar_vfe()
        self.pfe = PillarVFE(pillar_cfg, self.pillar_vfe["NUM_POINT_FEATURES"],
                             self.pillar_vfe["VOXEL_SIZE"], device).to(device)  # extract pillar's features
        self.psct = PointPillarScatter(pillar_cfg["NUM_FILTERS"][-1], pillar_vfe["GRID_SIZE"])  # map pillar to bev
        self.cnn = torchvision.models.resnet152(pretrained=True)
        # self.cnn.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, source_dict):
        # get features
        source_dict = self.pfe(source_dict, self.pillar_vfe["POINT_CLOUD_RANGE"])
        # map to bev
        source_dict = self.psct(source_dict)
        sbev = source_dict["spatial_features"]  # (B,C,H,W)
        out = self.cnn(sbev)
        return out
