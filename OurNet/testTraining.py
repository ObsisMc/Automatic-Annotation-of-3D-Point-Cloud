from models import SiameseNet
from dataset.NewDataSet import NewDataSet
from visualization.tensorboard import TensorBoardVis
from models.detector.NetPractice1 import PointNetPred
from utils import io_utils

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

blue = lambda x: '\033[94m' + x + '\033[0m'

root = "/home/zrh/Data/kitti/tracking/extracted_points"


def main(epochs=200, batch=1, shuffle=False, wokers=4, cudan=0):
    device = "cuda:%d" % cudan if torch.cuda.is_available() else "cpu"

    dataset = NewDataSet(root)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)) - 1,
                                                                           len(dataset) - int(0.8 * len(dataset)) + 1])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)
    valid_dataset = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)

    print(blue('# of training samples: %d' % len(train_dataset)))
    print(blue('# of validation samples: %d' % len(valid_dataset)))

    net = SiameseNet.Siamese2c()
    # net = PointNetPred(5)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    vis = TensorBoardVis()

    totalstep = 0
    for epoch in range(epochs):
        n = 0
        for i, data in enumerate(train_dataloader):
            points, label = data
            source = points[0].type(torch.FloatTensor)  # 一维卷积是在最后维度上扫的, change value in other dimension
            target = points[1].type(torch.FloatTensor)

            source, target, label = source.to(device), target.to(device), label.to(device)
            optimizer.zero_grad()
            net = net.train()
            pred = net(source, target)

            pred_loc, pred_angle, pred_cfd = pred[:, :3].to(device), pred[:, 3].to(device), pred[:, 4].to(device)

            loss_confidence = F.binary_cross_entropy_with_logits(pred_cfd, label[:, 4])
            loss_loc = F.smooth_l1_loss(pred_loc, label[:, :3])
            loss_angel = F.smooth_l1_loss(pred_angle, label[:, 3])  # maybe loss for angle can change to another one
            loss = 2 * loss_loc + 1 * loss_confidence + 1 * loss_angel

            loss.backward()
            optimizer.step()
            vis.add_scalar("loss_loc", loss_loc, totalstep)
            vis.add_scalar("loss_angle", loss_angel, totalstep)
            vis.add_scalar("loss_confidence", loss_confidence, totalstep)
            vis.add_scalar("loss", loss, totalstep)
            totalstep += 1
            n += 1
            if n % 100 == 0:
                print(loss)
                pass
        vis.add_scalar("total_loss", loss, epoch)
        scheduler.step()

        if epoch % 10 == 0:
            io_utils.saveCheckPoint(net, epoch, loss)


if __name__ == "__main__":
    main()
