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
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)  # 4d needs BatchNorm2d
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        # numbe of voxels
        if inputs.shape[1] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[1] // self.part
            part_linear_out = [self.linear(inputs[:, num_part * self.part:(num_part + 1) * self.part])
                               for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=1)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        # (batch, voxel_num, point_num, channel)
        x = self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=2, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, 1, inputs.shape[2], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=3)  # ? why it can work
            return x_concatenated


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, pillar_cfg, device):
        super().__init__()

        self.use_norm = model_cfg["USE_NORM"]
        self.with_distance = model_cfg["WITH_DISTANCE"]
        self.use_absolute_xyz = model_cfg["USE_ABSLOTE_XYZ"]
        self.num_filters = model_cfg["NUM_FILTERS"]

        self.num_point_features = pillar_cfg["NUM_POINT_FEATURES"] + 6 if self.use_absolute_xyz else 3
        self.voxel_size = pillar_cfg["VOXEL_SIZE"]
        self.point_cloud_range = pillar_cfg["POINT_CLOUD_RANGE"]

        self.device = device
        assert len(self.num_filters) > 0

        if self.with_distance:
            self.num_point_features += 1
        num_filters = [self.num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2)).to(self.device)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = self.voxel_size[0]
        self.voxel_y = self.voxel_size[1]
        self.voxel_z = self.voxel_size[2]
        self.x_offset = self.voxel_x / 2 + self.point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + self.point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + self.point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num):
        actual_num = torch.unsqueeze(actual_num, 2)

        max_num_shape = [1, -1]
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        max_num = torch.unsqueeze(max_num, 0).repeat(actual_num.shape[0], 1, 1)

        # since valid points are always in the front of array, the method can filter invalid points
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict):
        """
                ( our net doesn't use refraction so dimension corresponding to it doesn't exist )
                batch_dict:  why batch_index is needed? how to handle batch? how to train the network?
                !!!! following doesn't take batch into consideration!!!
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
        """
         our voxel_coords: (M, 3) -> (z, y, x)
        """
        # voxel_coords should add a dimension
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
        points_mean = voxel_features[:, :, :, :3].sum(dim=2, keepdim=True) / voxel_num_points.type_as(
            voxel_features).view(
            -1, voxel_num_points.shape[1], 1, 1)
        f_cluster = voxel_features[:, :, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :, :3])
        f_center[:, :, :, 0] = voxel_features[:, :, :, 0] - (
                coords[:, :, 2].to(voxel_features.dtype).unsqueeze(2) * self.voxel_x + self.x_offset)
        f_center[:, :, :, 1] = voxel_features[:, :, :, 1] - (
                coords[:, :, 1].to(voxel_features.dtype).unsqueeze(2) * self.voxel_y + self.y_offset)
        f_center[:, :, :, 2] = voxel_features[:, :, :, 2] - (
                coords[:, :, 0].to(voxel_features.dtype).unsqueeze(2) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)  # what happened
        # ignore invalid points
        point_count = features.shape[2]
        mask = self.get_paddings_indicator(voxel_num_points, point_count)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        # (batch, voxel_num, point_num, channel)
        for pfn in self.pfn_layers:
            features = pfn(features.to(self.device))
        features = features.squeeze(2)  # (batch, voxel_num, channel)
        batch_dict['pillar_features'] = features.permute(0, 2, 1)  # (batch, channel, voxel_num )
        return batch_dict


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
