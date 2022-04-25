from __future__ import print_function

import argparse
import os
import random
"""
Going to be obsolete
"""

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from OurNet.dataset.MyDataSet import MyDataSet
from MyModel import PointNetCls, PointNetPred
from OurNet.visualization.Visualize import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=1, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='../checkpoints',
                    help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str,
                    default='../Data/Mydataset/training',
                    help="dataset path")
parser.add_argument('--device', type=int, default=1, help="0 for cpu, 1 for gpu")
parser.add_argument('--ncard', type=int, default=0, help="serial number of gpu you want to use")


def main():
    opt = parser.parse_args()
    visualizer = Visualizer(opt.batchSize)
    device = "cuda:{}".format(opt.ncard) if opt.device else "cpu"
    print(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = MyDataSet()
    # Dataset is divided into training set and validation set.
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
        generator=torch.Generator().manual_seed(opt.manualSeed))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers))

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(blue('# of training samples: %d' % len(train_dataset)))
    print(blue('# of validation samples: %d' % len(valid_dataset)))
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    classifier = PointNetCls()
    predictor = PointNetPred()

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer1 = optim.Adam(classifier.parameters(), lr=0.001)
    optimizer2 = optim.Adam(predictor.parameters(), lr=0.001)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.5)
    classifier.to(device)
    predictor.to(device)

    num_batch = len(train_dataset) / opt.batchSize
    min_loss = 1e10
    for epoch in range(opt.nepoch):
        total_loss1 = 0
        total_loss1_1 = 0
        total_loss1_2 = 0
        total_loss2 = 0
        for i, data in enumerate(train_dataloader, 0):
            points, target, frame = data
            points1 = points[0].transpose(2, 1)
            points2 = points[1].transpose(2, 1)
            target = torch.tensor(target, dtype=torch.float32)
            points1 = points1.type(torch.FloatTensor)
            points2 = points2.type(torch.FloatTensor)
            points1, points2, target = points1.to(device), points2.to(device), target.to(device)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            classifier = classifier.train()  # todo: 不用sigmoid吗
            pred1 = classifier(points1, points2)
            # print("pred1: ", pred1.item())
            predictor = predictor.train()
            pred2 = predictor(points1, points2)
            target_cls = target[4].unsqueeze(0).unsqueeze(0)
            # print("target_cls: ", target_cls)
            loss1 = F.binary_cross_entropy_with_logits(pred1, target_cls).to(torch.float32)
            # print(loss1.item())
            loss2 = F.smooth_l1_loss(pred2, target[:4].unsqueeze(0).to(torch.float32)) * 30
            # if the actual value of target[4] is 0, then the loss2 is 0
            loss2 = loss2 * (target[4].to(torch.long) != 0).float()
            loss = loss1 + loss2
            if target[4].to(torch.long) != 0:
                loss.backward()
                optimizer1.step()
                optimizer2.step()
            else:
                loss1.backward()
                optimizer1.step()
            total_loss1_1 += loss1.item()
            total_loss1_2 += loss2.item()
            total_loss1 += loss.item()

            vpred = pred2.to("cpu").detach().numpy().tolist()[0] + pred1.to("cpu").detach().numpy().tolist()[0]
            vtarget = target.to("cpu").detach().numpy().tolist()
            vpoints1 = points1.to("cpu").detach().numpy()[0].T  # 第一维是batch
            vpoints2 = points2.to("cpu").detach().numpy()[0].T  # 第一维是batch
            visualizer.tablelog(vtarget, vpred, vpoints1, vpoints2, frame=int(frame[0]))
            if i % 100 == 0:
                visualizer.log(["cls loss (real time)"], [loss1])
                if target[4].to(torch.long) != 0:
                    visualizer.log(["adjustment loss (real time)"], [loss2])
                visualizer.log(["total loss (real time)"], [loss])
        total_loss1_1 /= len(train_dataset)
        total_loss1_2 /= len(train_dataset)
        total_loss1 /= len(train_dataset)
        print('train: epoch %d, loss1: %f, loss2: %f, average loss: %f' % (
            epoch, total_loss1_1, total_loss1_2, total_loss1))
        scheduler1.step()
        scheduler2.step()
        tp, fp, fn, tn = 0, 0, 0, 0
        visualizer.finishtable("table{}".format(epoch))
        for j, data in enumerate(valid_dataloader, 0):
            points, target, frame = data
            points1 = points[0].transpose(2, 1)
            points2 = points[1].transpose(2, 1)
            target = torch.tensor(target, dtype=torch.float32)
            points1 = points1.type(torch.FloatTensor)
            points2 = points2.type(torch.FloatTensor)
            points1, points2, target = points1.to(device), points2.to(device), target.to(device)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            classifier = classifier.train()
            pred1 = classifier(points1, points2)
            predictor = predictor.train()
            pred2 = predictor(points1, points2)
            if pred1.item() > 0.5 and target[4].to(torch.long) == 1:
                tp += 1
            if pred1.item() < 0.5 and target[4].to(torch.long) == 0:
                tn += 1
            if pred1.item() > 0.5 and target[4].to(torch.long) == 0:
                fp += 1
            if pred1.item() < 0.5 and target[4].to(torch.long) == 1:
                fn += 1
            target_cls = target[4].unsqueeze(0).unsqueeze(0)
            loss1 = F.binary_cross_entropy_with_logits(pred1, target_cls).to(torch.float32)
            loss2 = F.smooth_l1_loss(pred2, target[:4].unsqueeze(0).to(torch.float32)) * 30
            # if the actual value of target[4] is 0, then the loss2 is 0
            loss2 = loss2 * (target[4].to(torch.long) != 0).float()
            loss = loss1 + loss2
            total_loss2 += loss.item()
        accu = (tp + tn) / len(valid_dataset)
        paccu = tp / (tp + fp) if tp + fp > 0 else np.Inf
        naccu = tn / (tn + fn) if tn + fn > 0 else np.Inf
        recall = tp / (tp + fn) if tp + fn > 0 else np.Inf
        specificity = tn / (tn + fp) if tn + fp > 0 else np.Inf
        total_loss2 /= len(valid_dataset)
        print(blue('test: epoch %d, average loss: %f, accuracy: %f' % (epoch, total_loss2, accu)))
        # with open(opt.outf + '/log.txt', 'a') as f:
        #     f.write('epoch: %d, loss: %f\n' % (epoch, total_loss2))

        visualizer.log(["cls accuracy", "adjustment average loss",
                        "positive accuracy", "recall",
                        "negative accuracy", "specificity"],
                       [accu, total_loss2, paccu, recall, naccu, specificity])

        if min_loss > total_loss1:
            min_loss = total_loss1
            if epoch >= 10:
                torch.save(classifier.state_dict(), '%s/model_%d_%f.pth' % (opt.outf, epoch, total_loss1))


if __name__ == "__main__":
    main()
