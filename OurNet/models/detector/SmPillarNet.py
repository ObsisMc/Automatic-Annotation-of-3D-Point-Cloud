import torch
import torch.nn as nn
import torch.nn.functional as F

import common_utils.cfgs as Config
from OurNet.models.backbone3D.vfe.pillar_vfe import PillarVFE
from OurNet.models.backbone2D.pointpillar_scatter import PointPillarScatter


class SmPillarNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        pillar_cfg = Config.load_train_pillar_cfg()
        pillar_vfe = Config.load_pillar_vfe()
        self.pfe = PillarVFE(pillar_cfg, pillar_vfe, device).to(device)  # extract pillar's features
        self.psct = PointPillarScatter(pillar_cfg["NUM_FILTERS"][-1], pillar_vfe["GRID_SIZE"])  # map pillar to bev
        pass

    def forward(self, source_dict, target_dict):
        # get features
        source_dict, target_dict = self.pfe(source_dict), self.pfe(target_dict)
        # map to bev
        source_dict, target_dict = self.psct(source_dict), self.psct(target_dict)
        sbev, tbev = source_dict["spatial_features"], target_dict["spatial_features"]  # (B,C,H,W)


        return sbev
