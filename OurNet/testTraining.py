from testNet import PointNetCls as Net
from dataset.NewDataSet import NewDataSet
from models.NewNet import NewNet

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

blue = lambda x: '\033[94m' + x + '\033[0m'

root = "/home/zrh/Data/kitti/tracking/extracted_points"


def main(epochs=10, batch=1, shuffle=False, wokers=4, cudan=0):
    device = "cuda:%d" % cudan if torch.cuda.is_available() else "cpu"

    dataset = NewDataSet(root)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)),
                                                                           len(dataset) - int(0.8 * len(dataset))])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)
    valid_dataset = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)

    print(blue('# of training samples: %d' % len(train_dataset)))
    print(blue('# of validation samples: %d' % len(valid_dataset)))

    net = NewNet()
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(epochs):
        n = 0
        for i, data in enumerate(train_dataloader):
            source, target, label = data[0][0].to(device), data[0][1].to(device), data[1].to(device)
            source = source.permute(0, 2, 1).type(torch.FloatTensor)  # 一维卷积是在最后维度上扫的, change value in other dimension
            target = target.permute(0, 2, 1).type(torch.FloatTensor)

            optimizer.zero_grad()
            pred_tran, pred_angle, pred_cfd = net(source, target)

            loss_confidence = F.binary_cross_entropy_with_logits(pred_cfd.to(device), label[:, 4])
            loss_loc = F.smooth_l1_loss(pred_tran.to(device), label[:, :3])
            loss_angel = F.smooth_l1_loss(pred_angle.to(device),
                                          label[:, 3])  # maybe loss for angle can change to another one
            loss = 2 * loss_loc + loss_confidence + 0.2 * loss_angel
            loss.backward()

            optimizer.step()
            scheduler1.step()
            n += 1
            if n % 100 == 0:
                print(loss)
                pass


if __name__ == "__main__":
    main()
