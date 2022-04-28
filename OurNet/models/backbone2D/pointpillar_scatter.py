import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    def __init__(self, num_bev_feature, grid_size, **kwargs):
        super().__init__()

        self.num_bev_features = num_bev_feature
        self.nx, self.ny, self.nz = grid_size.astype(int)
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        """
        pillar_features: (B,C,M), C -> channel,  M -> pillars' number
        voxel_coords: (B,M,3), 3 -> (z,y,x)
        """
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = pillar_features.shape[0]
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)  # (C,H*W)

            # TODO: not sure why need batch_index, is it unordered? need more test
            # batch_mask = coords[:, 0] == batch_idx
            # this_coords = coords[batch_mask, :]

            this_coords = coords[batch_idx, :]  # (M,3)
            indices = this_coords[:, 0] + this_coords[:, 1] * self.nx + this_coords[:, 2]  # flatten img & find the pos
            indices = indices.type(torch.long)  # (M,)
            # pillars = pillar_features[batch_mask, :]
            pillars = pillar_features[batch_idx, :]  # (C,M)

            spatial_feature[:, indices] = pillars  # scatter pillars into bev
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny,
                                                             self.nx)
        batch_dict['spatial_features'] = batch_spatial_features  # (B,C,H,W)
        return batch_dict
