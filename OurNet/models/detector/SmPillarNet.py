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
        self.pfe = PillarVFE(pillar_cfg, pillar_vfe, device).to(device)  # incise pillars with feature extracted
        self.psct = PointPillarScatter(pillar_cfg["NUM_FILTERS"][-1], pillar_vfe["GRID_SIZE"])  # map pillar to bev
        pass

    def forward(self, source_dict, target_dict):
        source_dict = self.pfe(source_dict)
        source_dict = self.psct(source_dict)
        bev = source_dict["spatial_features"]  # (B,C,H,W)
        return bev
