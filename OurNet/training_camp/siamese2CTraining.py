from OurNet.models.detector.SiameseNet import Siamese2c
from OurNet.dataset.NewDataSet import NewDataSet
from OurNet.visualization.tensorboard import TensorBoardVis
from OurNet.models.model_utils import io_utils

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

blue = lambda x: '\033[94m' + x + '\033[0m'


def main(epochs=200, batch=5, shuffle=False, wokers=4, cudan=0):
    device = "cuda:%d" % cudan if torch.cuda.is_available() else "cpu"

    dataset = NewDataSet()
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)) - 1,
                                                                           len(dataset) - int(0.8 * len(dataset)) + 1])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)
    valid_dataset = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)

    print(blue('# of training samples: %d' % len(train_dataset)))
    print(blue('# of validation samples: %d' % len(valid_dataset)))

    net = Siamese2c()
    # net = PointNetPred(5)
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    vis = TensorBoardVis(net=net)

    totalstep = 0
    for epoch in range(epochs):
        n = 0
        for i, data in enumerate(train_dataloader):
            points, label = data
            source = points[0].to(torch.float32)  # 一维卷积是在最后维度上扫的, change value in other dimension
            target = points[1].to(torch.float32)
            label = label.to(torch.float32)

            source, target, label = source.to(device), target.to(device), label.to(device)
            optimizer.zero_grad()
            net = net.train()
            pred = net(source, target).to(torch.float32)

            mask = label[:, 4] == 1
            batch_idx = torch.arange(source.shape[0])
            pos_pred = pred[batch_idx[mask], :].to(device)
            pos_label = label[batch_idx[mask], :].to(device)
            neg_pred = pred[batch_idx[mask == False], :].to(device)
            neg_label = label[batch_idx[mask == False], :].to(device)

            pos_loss = 0
            if pos_pred.shape[0] != 0:
                pos_loss_confidence = F.binary_cross_entropy_with_logits(pos_pred[:, 4], pos_label[:, 4])
                loss_loc = F.smooth_l1_loss(pos_pred[:, :3], pos_label[:, :3])
                loss_angel = F.smooth_l1_loss(pos_pred[:, 3], pos_label[:, 3])
                pos_loss = 2 * loss_loc + 1 * pos_loss_confidence + 1 * loss_angel
                vis.add_scalar("loss_loc", loss_loc, totalstep)
                vis.add_scalar("loss_angle", loss_angel, totalstep)
                vis.add_scalar("pos_loss_confidence", pos_loss_confidence, totalstep)
                vis.add_scalar("pos_loss", pos_loss, totalstep)

            neg_loss = 0
            if neg_pred.shape[0] != 0:
                neg_loss_confidence = F.binary_cross_entropy_with_logits(neg_pred[:, 4], neg_label[:, 4])
                neg_loss = 4 * neg_loss_confidence
                vis.add_scalar("neg_loss_confidence", neg_loss_confidence, totalstep)
                vis.add_scalar("neg_loss", neg_loss, totalstep)

            loss = pos_loss + neg_loss
            loss = loss.to(torch.float32)
            loss.backward()
            optimizer.step()

            vis.add_scalar("loss", loss, totalstep)
            totalstep += 1
            n += 1
            if n % 100 == 0:
                print("Epoch %d. total loss is %f " %
                      (epoch, loss))
                pass
        vis.add_scalar("total_loss", loss, epoch)
        scheduler.step()

        if epoch % 10 == 0 and epoch > 0:
            io_utils.saveCheckPoint(net, epoch, loss)


if __name__ == "__main__":
    main()
