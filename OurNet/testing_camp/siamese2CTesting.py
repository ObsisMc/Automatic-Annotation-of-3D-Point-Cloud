import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
import os
import numpy as np

import common_utils.visual_utils.visual_modul.open3d_vis_utils as o3d_utils
from OurNet.models.detector.SiameseNet import Siamese2c, SiameseAttentionMulti, SiameseMultiDecoder
from OurNet.dataset.dataset_utils.Sampler import Sampler
from OurNet.dataset.NewDataSet import NewDataSet

blue = lambda x: '\033[94m' + x + '\033[0m'


def test_Siamese2c():
    print()
    source_path = "/home/zrh/Data/kitti/tracking/extracted_points_entend13/0000/Van#0/points/000000.npy"
    target_path = "/home/zrh/Data/kitti/tracking/extracted_points_entend13/0000/Van#0/points/000001.npy"

    sampler = Sampler()
    raw_source = sampler.fps(np.load(source_path))
    raw_target = sampler.fps(np.load(target_path))
    target = o3d_utils.rotate_points_along_z(raw_target, 0.4)
    target = target + np.array([0.2, 0.2, 0]).reshape(1, -1)

    # o3d_utils.draw_object(points=target)

    source_tensor = torch.tensor(raw_source, dtype=torch.float32).unsqueeze(0)
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

    # o3d_utils.draw_object(points=None, multi_points=[raw_source, target], colorful=True)

    net = Siamese2c()
    ckp_pth = "/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech/OurNet/checkpoints/Siamese2c/ckpt_epc120_0.028655.pth"
    state_dict = torch.load(ckp_pth)
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
    ckpt = "/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech/OurNet/checkpoints/SiameseAttentionMulti/ckpt_epc80_0.029859.pth"
    state_dict = torch.load(ckpt)
    net.load_state_dict(state_dict)
    pred = net(source_tensor, target_tensor)

    dx, dy, dz = pred[0][0, 0].detach(), pred[1][0, 0].detach(), pred[2][0, 0].detach()
    translation = np.array([dx, dy, dz])
    angel = pred[3][0, 0].detach()

    print(translation, angel)

    # target_adjust = o3d_utils.rotate_points_along_z(target, angel) + translation
    # o3d_utils.draw_object(points=None, multi_points=[raw_source, target_adjust], colorful=True)


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
    ckpt = "/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech/OurNet/checkpoints/SiameseMultiDecoder/ckpt_epc20_0.088397.pth"
    state_dict = torch.load(ckpt)
    net.load_state_dict(state_dict)
    pred = net(source_tensor, target_tensor)

    dx, dy, dz = pred[0][0, 0].detach(), pred[1][0, 0].detach(), pred[2][0, 0].detach()
    translation = np.array([dx, dy, dz])
    angel = pred[3][0, 0].detach()

    print(translation, angel)

    target_adjust = o3d_utils.rotate_points_along_z(target, angel) + translation
    o3d_utils.draw_object(points=None, multi_points=[raw_source, target_adjust], colorful=True)


def eval(batchs=1, workers=4, shuffle=False):
    def init_dataset_net(model: int, device: str):
        dataset_tmp = net_tmp = None
        ckpt_root = "/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech/OurNet/checkpoints/"
        if model == 0:  # Siamese2C
            ckp_pth = os.path.join(ckpt_root, "Siamese2c/ckpt_epc120_0.028655.pth")
            dataset_tmp = NewDataSet("/home/zrh/Data/kitti/tracking/extracted_points_default")
            net_tmp = Siamese2c().to(device)
            net_tmp.load_state_dict(torch.load(ckp_pth))
        elif model == 1:  # SiameseMultiDecoder
            ckp_pth = os.path.join(ckpt_root, "SiameseMultiDecoder/ckpt_epc20_0.088397.pth")
            dataset_tmp = NewDataSet("/home/zrh/Data/kitti/tracking/extracted_points_default")
            net_tmp = SiameseMultiDecoder().to(device)
            net_tmp.load_state_dict(torch.load(ckp_pth))
        elif model == 2:  # SiameseAttentionMulti
            ckp_pth = os.path.join(ckpt_root, "SiameseAttentionMulti/ckpt_epc80_0.029859.pth")
            dataset_tmp = NewDataSet("/home/zrh/Data/kitti/tracking/extracted_points_default")
            net_tmp = SiameseAttentionMulti().to(device)
            net_tmp.load_state_dict(torch.load(ckp_pth))
        assert dataset_tmp and net_tmp
        return dataset_tmp, net_tmp

    def calc_error(lb: torch.Tensor, pd: torch.Tensor):
        loss_x = torch.sum(torch.abs(lb[0] - pd[0])).detach()
        loss_y = torch.sum(torch.abs(lb[1] - pd[1])).detach()
        loss_z = torch.sum(torch.abs(lb[2] - pd[2])).detach()
        loss_angel = torch.sum(torch.abs(lb[3] - pd[3])).detach()
        loss_confidence = F.binary_cross_entropy_with_logits(pd[4], lb[4]).detach()
        return loss_x, loss_y, loss_z, loss_angel, loss_confidence

    batchs, workers, shuffle = batchs, workers, shuffle
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset, net = init_dataset_net(0, device)
    _, valid_data = torch.utils.data.random_split(dataset, [0, len(dataset)])
    vaild_dataloader = torch.utils.data.DataLoader(valid_data, num_workers=workers, batch_size=batchs, shuffle=shuffle)

    valid_len = len(valid_data)
    print(blue('# of validation samples: %d' % valid_len))

    error_x = error_y = error_z = error_angel = error_con = 0
    for i, data in enumerate(vaild_dataloader):
        net.eval()
        points, label = data
        source = points[0].type(torch.FloatTensor)
        target = points[1].type(torch.FloatTensor)
        source, target, label = source.to(device), target.to(device), label.to(device)

        pred = net(source, target)
        pred = pred.squeeze()
        label = label.squeeze()

        loss_x, loss_y, loss_z, loss_angel, loss_confidence = calc_error(label, pred)
        error_x += loss_x
        error_y += loss_y
        error_z += loss_z
        error_angel += loss_angel
        error_con += loss_confidence
    avg_e_x = error_x / valid_len
    avg_e_y = error_y / valid_len
    avg_e_z = error_z / valid_len
    avg_e_angel = error_angel / valid_len
    avg_e_con = error_con / valid_len
    print("error_x: %f, error_y: %f, error_z: %f, error_angel: %f, error_con: %f" %
          (avg_e_x, avg_e_y, avg_e_z, avg_e_angel, avg_e_con))


if __name__ == "__main__":
    eval()
