from OurNet.models.detector.SiameseNet import SiamesePlus
from OurNet.dataset.NewDataSet import NewDataSet
from OurNet.visualization.tensorboard import TensorBoardVis
from OurNet.models.model_utils import io_utils

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

blue = lambda x: '\033[94m' + x + '\033[0m'


def main(epochs=50, batch=5, shuffle=False, wokers=4, cudan=0):
    device = "cuda:%d" % cudan if torch.cuda.is_available() else "cpu"

    dataset = NewDataSet("/home/zrh/Data/kitti/tracking/extracted_points_entend13")
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)) - 1,
                                                                           len(dataset) - int(0.8 * len(dataset)) + 1])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)
    valid_dataset = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)

    print(blue('# of training samples: %d' % len(train_dataset)))
    print(blue('# of validation samples: %d' % len(valid_dataset)))

    net = SiamesePlus()
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    vis = TensorBoardVis(net=net)

    totalstep = 0
    for epoch in range(epochs):
        n = 0
        for i, data in enumerate(train_dataloader):
            points, label = data
            source = points[0].type(torch.FloatTensor)  # 一维卷积是在最后维度上扫的, change value in other dimension
            target = points[1].type(torch.FloatTensor)
            label = label.to(torch.float32)

            source, target, label = source.to(device), target.to(device), label.to(device)
            optimizer.zero_grad()
            net = net.train()
            pred = net(source, target)
            loss = F.smooth_l1_loss(pred, label[:, 0].view(-1, 1))
            loss.backward()
            optimizer.step()
            if n % 100 == 0:
                print("Training loss: %f" % loss.detach())
            n += 1
        scheduler.step()
        total_loss = 0
        for i, data in enumerate(valid_dataset):
            points, label = data
            source = points[0].type(torch.FloatTensor)  # 一维卷积是在最后维度上扫的, change value in other dimension
            target = points[1].type(torch.FloatTensor)

            source, target, label = source.to(device), target.to(device), label.to(device)
            optimizer.zero_grad()
            net = net.train()
            pred = net(source, target)
            loss = F.smooth_l1_loss(pred, label[:, 0].view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
        print("Validation loss: %f" % (total_loss / len(valid_dataset)))


if __name__ == "__main__":
    main()
