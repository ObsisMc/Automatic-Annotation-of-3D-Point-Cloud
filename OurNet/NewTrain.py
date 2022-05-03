from __future__ import print_function


import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from OurNet.models.model_utils import io_utils

from dataset.NewDataSet import NewDataSet
from NewModel import NewModel
from visualization.tensorboard import TensorBoardVis
import random



blue = lambda x: '\033[94m' + x + '\033[0m'
device = "cuda:%d" % 0 if torch.cuda.is_available() else "cpu"
# print(device)

def main(epochs=200, batch=5, shuffle=False, workers=4):

    dataset = NewDataSet(io_utils.getDataSetPath())
    manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
        generator=torch.Generator().manual_seed(manualSeed))


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers)

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers)

    print(blue('# of training samples: %d' % len(train_dataset)))
    print(blue('# of validation samples: %d' % len(valid_dataset)))


    Net = NewModel()


    optimizer = optim.Adam(Net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    vis = TensorBoardVis()
    Net.to(device)

    totalstep = 0
    for epoch in range(epochs):
        n = 0
        for i, data in enumerate(train_dataloader, 0):
            # print(data)
            points, label = data
            points1 = points[0].type(torch.FloatTensor)
            points2 = points[1].type(torch.FloatTensor)

            points1, points2, label = points1.to(device), points2.to(device), label.to(device)
            optimizer.zero_grad()
            Net = Net.train()
            pred = Net(points1, points2)

            pred_loc, pred_angle, pred_cfd = pred[:, :3].to(device), pred[:, 3].to(device), pred[:, 4].to(device)

            loss_confidence = F.binary_cross_entropy_with_logits(pred_cfd, label[:, 4])
            loss_loc = F.smooth_l1_loss(pred_loc, label[:, :3])
            loss_angel = F.smooth_l1_loss(pred_angle, label[:, 3])
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

        if epoch % 10 == 0 and epoch > 0:
            io_utils.saveCheckPoint(Net, epoch, loss)

if __name__ == "__main__":
    main()
