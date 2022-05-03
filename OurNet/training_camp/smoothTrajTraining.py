import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import common_utils.cfgs as Config
from OurNet.visualization.tensorboard import TensorBoardVis
from OurNet.models.model_utils import io_utils

from OurNet.dataset.SmoothTrajDataSet import SmoothTrajDataSet
from OurNet.models.detector.SmoothTrajNet import SmoothTrajNet

blue = lambda x: '\033[94m' + x + '\033[0m'


def main(train=True):
    vis = TensorBoardVis()
    train_cfg = Config.load_train_common()
    train_para = train_cfg["training_parameters"]
    batchs, workers, shuffle = train_para["batch_size"], train_para["workers"], train_para["shuffle"]
    epochs = train_para["epochs"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    max_traj_n = 10
    dataset = SmoothTrajDataSet(train_cfg["dataset_path"], max_traj_n=max_traj_n)
    train_num = int(len(dataset) * train_para["train_ratio"])
    train_data, valid_data = torch.utils.data.random_split(dataset, [train_num, len(dataset) - train_num])
    train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=workers, batch_size=batchs, shuffle=shuffle)
    vaild_dataloader = torch.utils.data.DataLoader(train_data, num_workers=workers, batch_size=batchs, shuffle=shuffle)

    net = SmoothTrajNet(device, N=max_traj_n).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_len = len(train_data)
    print(blue('# of training samples: %d' % train_len))
    print(blue('# of validation samples: %d' % len(valid_data)))

    if train:
        n = 0
        base = 5
        for epoch in range(epochs):
            for i, data in enumerate(train_dataloader):
                net = net.train()
                point_dicts, poses, labels = data
                poses = poses.to(device)
                pred = net(point_dicts, poses)

                pred = pred.view(batchs, -1, 3).to(device)
                labels = labels.to(device)

                # angle_diff = torch.sin(pred[:, :, 2] - labels[:, :, 2])
                loss_dx = F.smooth_l1_loss(pred[:, :, 0], labels[:, :, 0])
                loss_dy = F.smooth_l1_loss(pred[:, :, 1], labels[:, :, 1])
                loss_angle = F.smooth_l1_loss(pred[:, :, 2], labels[:, :, 2])

                loss = 2 * (loss_dy + loss_dx + loss_angle)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                vis.add_scalar("loss_x", loss_dx, n)
                vis.add_scalar("loss_y", loss_dy, n)
                vis.add_scalar("loss_angle", loss_angle, n)
                vis.add_scalar("loss_total", loss, n)
                n += 1
                if n % 100 == 0:
                    print("Process %d/%d (Epoch %d) --> Loss: %f x:%f, y:%f, angle:%f" % (
                        i, train_len, epoch, loss, float(loss_dx), float(loss_dy), float(loss_angle)))
            scheduler.step()
            valid_loss = 0
            for i, data in enumerate(vaild_dataloader):
                net = net.eval()
                point_dicts, poses, labels = data
                poses = poses.to(device)
                pred = net(point_dicts, poses)

                pred = pred.view(batchs, -1, 3).to(device)
                labels = labels.to(device)

                # angle_diff = torch.sin(pred[:, :, 2] - labels[:, :, 2])
                loss_dx = F.smooth_l1_loss(pred[:, :, 0], labels[:, :, 0])
                loss_dy = F.smooth_l1_loss(pred[:, :, 1], labels[:, :, 1])
                loss_angle = F.smooth_l1_loss(pred[:, :, 2], labels[:, :, 2])

                valid_loss = 2 * loss_dy + 2 * loss_dx + 2 * loss_angle
                vis.add_scalar("loss_total_valid", valid_loss, epoch)
            if epoch % 10 == 0:
                io_utils.saveCheckPoint(net, epoch, valid_loss)
    else:
        ckpt = "/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech/OurNet/checkpoints/SmoothTrajNet/ckpt_epc0_0.004273.pth"
        test(net, ckpt, train_dataloader, device, batchs)


def test(net: torch.nn.Module, ckpt_path, dataloader, device, batchs):
    net.load_state_dict(torch.load(ckpt_path))
    for i, data in enumerate(dataloader):
        net.eval()
        point_dicts, poses, labels = data
        poses = poses.to(device)
        pred = net(point_dicts, poses)

        pred = pred.view(batchs, -1, 3).to(device)
        labels = labels.to(device)

        print(labels - pred)
        break


if __name__ == "__main__":
    main()
