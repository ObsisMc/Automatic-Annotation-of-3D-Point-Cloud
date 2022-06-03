import torch
import torch.nn

import numpy as np

import common_utils.visual_utils.visual_modul.open3d_vis_utils as o3d_utils
from OurNet.models.detector.SiameseNet import Siamese2c, SiameseAttentionMulti, SiameseMultiDecoder
from OurNet.dataset.dataset_utils.Sampler import Sampler


def test_Siamese2c():
    print()
    source_path = "/home/zrh/Data/kitti/tracking/extracted_points_entend13/0000/Van#0/points/000000.npy"
    target_path = "/home/zrh/Data/kitti/tracking/extracted_points_entend13/0000/Van#0/points/000001.npy"

    sampler = Sampler()
    raw_source = sampler.fps(np.load(source_path))
    raw_target = sampler.fps(np.load(target_path))
    target = o3d_utils.rotate_points_along_z(raw_target, 0.4)
    target = target + np.array([0.2, 0.1, 0]).reshape(1, -1)

    # o3d_utils.draw_object(points=target)

    source_tensor = torch.tensor(raw_source, dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

    # o3d_utils.draw_object(points=None, multi_points=[raw_source, target], colorful=True)

    net = Siamese2c()
    state_dict = torch.load(
        "/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech/OurNet/checkpoints/Siamese2c/ckpt_epc170_0.002786.pth")
    net.load_state_dict(state_dict)
    pred = net(source_tensor, target_tensor)

    print(pred)
    angel = pred[0, 3].detach()
    translation = pred[0, :3].detach().numpy().reshape(1, 3)
    target_adjust = o3d_utils.rotate_points_along_z(target, angel) + translation
    o3d_utils.draw_object(points=None, multi_points=[raw_source, target_adjust], colorful=True)


def test_SiameseAttentionMulti():
    print()
    source_path = "/home/zrh/Data/kitti/tracking/extracted_points_entend13/0000/Van#0/points/000000.npy"
    target_path = "/home/zrh/Data/kitti/tracking/extracted_points_entend13/0000/Van#0/points/000001.npy"

    sampler = Sampler()
    raw_source = sampler.fps(np.load(source_path))
    raw_target = sampler.fps(np.load(target_path))
    target = o3d_utils.rotate_points_along_z(raw_target, 0.1)
    target = target + np.array([0.2, 0.1, 0]).reshape(1, -1)

    # o3d_utils.draw_object(points=target)

    source_tensor = torch.tensor(raw_source, dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

    # o3d_utils.draw_object(points=None, multi_points=[raw_source, target], colorful=True)

    net = SiameseAttentionMulti()
    state_dict = torch.load(
        "/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech/OurNet/checkpoints/SiameseAttentionMulti/ckpt_epc80_0.029859.pth")
    net.load_state_dict(state_dict)
    pred = net(source_tensor, target_tensor)

    dx, dy, dz = pred[0][0, 0].detach(), pred[1][0, 0].detach(), pred[2][0, 0].detach()
    translation = np.array([dx, dy, dz])
    angel = pred[3][0, 0].detach()

    print(translation, angel)

    target_adjust = o3d_utils.rotate_points_along_z(target, angel) + translation
    o3d_utils.draw_object(points=None, multi_points=[raw_source, target_adjust], colorful=True)


def test_SiameseMultiDecoder():
    print()
    source_path = "/home/zrh/Data/kitti/tracking/extracted_points_entend13/0000/Van#0/points/000000.npy"
    target_path = "/home/zrh/Data/kitti/tracking/extracted_points_entend13/0000/Van#0/points/000001.npy"

    sampler = Sampler()
    raw_source = sampler.fps(np.load(source_path))
    raw_target = sampler.fps(np.load(target_path))
    target = o3d_utils.rotate_points_along_z(raw_target, 0.1)
    target = target + np.array([0.2, 0.1, 0]).reshape(1, -1)

    # o3d_utils.draw_object(points=target)

    source_tensor = torch.tensor(raw_source, dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

    # o3d_utils.draw_object(points=None, multi_points=[raw_source, target], colorful=True)

    net = SiameseMultiDecoder()
    state_dict = torch.load(
        "/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech/OurNet/checkpoints/SiameseMultiDecoder/ckpt_epc20_0.088397.pth")
    net.load_state_dict(state_dict)
    pred = net(source_tensor, target_tensor)

    dx, dy, dz = pred[0][0, 0].detach(), pred[1][0, 0].detach(), pred[2][0, 0].detach()
    translation = np.array([dx, dy, dz])
    angel = pred[3][0, 0].detach()

    print(translation, angel)

    target_adjust = o3d_utils.rotate_points_along_z(target, angel) + translation
    o3d_utils.draw_object(points=None, multi_points=[raw_source, target_adjust], colorful=True)
