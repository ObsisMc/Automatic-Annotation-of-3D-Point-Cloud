import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from OurNet.dataset.dataset_utils.processor import data_processor


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)  # ? why it can work
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__()

        self.use_norm = model_cfg.USE_NORM
        self.with_distance = model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def map_to_bev(self, batch_dict):
        """
        currently only support batch size 1
        """
        c = batch_dict["pillar_features"].shape[1]
        w = int(batch_dict["image_shape"][0])
        h = int(batch_dict["image_shape"][1])
        bev = torch.zeros((c, h, w))
        for index in range(len(batch_dict["voxel_coords"])):
            coo = batch_dict["voxel_coords"][index].type(torch.int)
            bev[:, coo[2], coo[3]] = batch_dict["pillar_features"][index]
        batch_dict["bev"] = bev
        return batch_dict

    def forward(self, batch_dict, **kwargs):
        """
                batch_dict:  why batch_index is needed? how to handle batch? how to train the network?
                    points:(N,5) --> (batch_index,x,y,z,r) batch_index代表了该点云数据在当前batch中的index
                    frame_id:(4,) --> (003877,001908,006616,005355) 帧ID
                    gt_boxes:(4,40,8)--> (x,y,z,dx,dy,dz,ry,class)
                    use_lead_xyz:(4,) --> (1,1,1,1)
                    voxels:(M,32,4) --> (x,y,z,r)
                    voxel_coords:(M,4) --> (batch_index,z,y,x) batch_index代表了该点云数据在当前batch中的index
                    voxel_num_points:(M,)
                    image_shape:(4,2) 每份点云数据对应的2号相机图片分辨率
                    batch_size:4    batch_size大小
        """
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(
            -1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)  # what happened

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze(1)  # zrh fix: add 1 in squeeze
        batch_dict['pillar_features'] = features
        return self.map_to_bev(batch_dict)


class Config():
    def __init__(self):
        self.USE_NORM = True
        self.WITH_DISTANCE = False
        self.USE_ABSLOTE_XYZ = True
        self.NUM_FILTERS = [64]


def handleDataDict(data_dict):
    data_dict['voxels'] = torch.Tensor(data_dict['voxels'])
    data_dict['voxel_num_points'] = torch.Tensor(data_dict['voxel_num_points'])
    data_dict['voxel_coords'] = torch.Tensor(
        np.c_[np.array([index for index in range(data_dict['voxel_coords'].shape[0])]).reshape(-1, 1), data_dict[
            'voxel_coords']]
    )


if __name__ == "__main__":
    data_dict, vrange = data_processor.getDataDict()
    handleDataDict(data_dict)
    config = Config()
    pvfe = PillarVFE(model_cfg=config, num_point_features=4, voxel_size=[0.16, 0.16, 4.0], point_cloud_range=vrange)
    pvfe.forward(data_dict)
    print(data_dict)
