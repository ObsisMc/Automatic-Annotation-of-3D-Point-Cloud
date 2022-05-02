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
    def __init__(self, model_cfg, num_point_features, voxel_size, device):
        super().__init__()

        self.use_norm = model_cfg["USE_NORM"]
        self.with_distance = model_cfg["WITH_DISTANCE"]
        self.use_absolute_xyz = model_cfg["USE_ABSLOTE_XYZ"]
        self.num_filters = model_cfg["NUM_FILTERS"]

        self.num_point_features = num_point_features + 6 if self.use_absolute_xyz else 3
        self.voxel_size = voxel_size

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
        # z should calc dynamic

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

    def forward(self, batch_dict, point_cloud_range):
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

        def augFeature(v_features, v_num_points, crds, pt_cld_rge):
            # (B, num_voxels, num_points, xyz), (B, num_voxels), (B,num_voxels,zyx), (B, [lx,ly,lz,ux,uy,uz])
            voxel_z = (pt_cld_rge[:, 5] - pt_cld_rge[:, 2]).view(-1, 1)  # (B,1)

            x_offset = (self.voxel_x / 2 + pt_cld_rge[:, 0]).view(-1, 1)  # (B,1)
            y_offset = (self.voxel_y / 2 + pt_cld_rge[:, 1]).view(-1, 1)  # (B,1)
            z_offset = (voxel_z / 2 + pt_cld_rge[:, 2]).view(-1, 1)  # (B,1)

            points_mean = v_features[:, :, :, :3].sum(dim=2, keepdim=True) / v_num_points.type_as(
                v_features).view(
                -1, v_num_points.shape[1], 1, 1)
            f_cluster = v_features[:, :, :, :3] - points_mean

            f_center = torch.zeros_like(v_features[:, :, :, :3])
            # f_center[:, :, :, 0] = v_features[:, :, :, 0] - (
            #         crds[:, :, 2].to(v_features.dtype).unsqueeze(2) * self.voxel_x + x_offset)
            # f_center[:, :, :, 1] = v_features[:, :, :, 1] - (
            #         crds[:, :, 1].to(v_features.dtype).unsqueeze(2) * self.voxel_y + y_offset)
            # f_center[:, :, :, 2] = v_features[:, :, :, 2] - (
            #         crds[:, :, 0].to(v_features.dtype).unsqueeze(2) * voxel_z + z_offset)
            f_center[:, :, :, 0] = v_features[:, :, :, 0] - (
                    crds[:, :, 2].to(v_features.dtype) * self.voxel_x + x_offset).unsqueeze(2)
            f_center[:, :, :, 1] = v_features[:, :, :, 1] - (
                    crds[:, :, 1].to(v_features.dtype) * self.voxel_y + y_offset).unsqueeze(2)
            f_center[:, :, :, 2] = v_features[:, :, :, 2] - (
                    crds[:, :, 0].to(v_features.dtype) * voxel_z + z_offset).unsqueeze(2)

            if self.use_absolute_xyz:
                features = [v_features, f_cluster, f_center]
            else:
                features = [v_features[..., 3:], f_cluster, f_center]

            if self.with_distance:
                points_dist = torch.norm(v_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            features = torch.cat(features, dim=-1)  # what happened
            return features

        # voxel_coords should add a dimension
        # (B, num_voxels, num_points, xyz), (B, num_voxels), (B,num_voxels,xyz)
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict[
            'voxel_coords']
        if type(point_cloud_range) is not torch.tensor:
            point_cloud_range = torch.tensor(point_cloud_range)
            if point_cloud_range.ndim == 1:
                point_cloud_range = point_cloud_range.view(1, -1).repeat(voxel_features.shape[0], 1)
        features = augFeature(voxel_features, voxel_num_points, coords, point_cloud_range)
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
