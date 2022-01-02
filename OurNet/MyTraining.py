from __future__ import print_function

import argparse
import os
import random

import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from MyDataSet import MyDataSet
from MyModel import PointNetCls

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=1, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='checkpoints',
                    help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str,
                    default='../Data/Mydataset/training',
                    help="dataset path")
parser.add_argument('--cpu', action="store_true", default=True)


def main():
    opt = parser.parse_args()
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
        shuffle=True,
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

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    if not opt.cpu:
        classifier.cuda()

    num_batch = len(dataset) / opt.batchSize

    for epoch in range(opt.nepoch):
        for i, data in enumerate(train_dataloader, 0):
            points, target = data
            points1 = points[0].transpose(2, 1)
            points2 = points[1].transpose(2, 1)
            target = torch.tensor(target, dtype=torch.float32)
            points1 = points1.type(torch.FloatTensor)
            points2 = points2.type(torch.FloatTensor)
            if not opt.cpu:
                points1, points2, target = points1.cuda(), points2.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred1, pred2 = classifier(points1, points2)
            target_cls = target[4].to(torch.long).unsqueeze(0)
            loss1 = F.cross_entropy(pred1, target_cls).to(torch.float32)
            loss2 = F.mse_loss(pred2, target[:4].unsqueeze(0).to(torch.float32))
            # if the actual value of target[4] is 0, then the loss2 is 0
            loss2 = loss2 * (target[4] != 0).float()
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            scheduler.step()
            print('[%d: %d/%d] train loss1: %f  loss2: %f  total loss: %f' % (
                epoch, i, num_batch, loss1.item(), loss2.item(), loss.item()))
            if i % 10 == 0:
                j, data = next(enumerate(valid_dataloader, 0))
                points, target = data
                points1 = points[0].transpose(2, 1)
                points2 = points[1].transpose(2, 1)
                target = torch.tensor(target, dtype=torch.float32)
                points1 = points1.type(torch.FloatTensor)
                points2 = points2.type(torch.FloatTensor)
                if not opt.cpu:
                    points1, points2, target = points1.cuda(), points2.cuda(), target.cuda()
                optimizer.zero_grad()
                classifier = classifier.train()
                pred1, pred2 = classifier(points1, points2)
                target_cls = target[4].to(torch.long).unsqueeze(0)
                loss1 = F.cross_entropy(pred1, target_cls).to(torch.float32)
                loss2 = F.mse_loss(pred2, target[:4].unsqueeze(0).to(torch.float32))
                loss2 = loss2 * (target[4] != 0).float()
                loss = loss1 + loss2
                with open(opt.outf + '/log.txt', 'a') as f:
                    f.write('[%d: %d/%d] loss1: %f  loss2: %f  total loss: %f\n' % (
                        epoch, i, num_batch, loss1.item(), loss2.item(), loss.item()))
                print('[%d: %d/%d] %s loss: %f' % (
                    epoch, i, num_batch, blue('test'), loss.item()))

        torch.save(classifier.state_dict(), '%s/model_%d.pth' % (opt.outf, epoch))

    # total_correct = 0
    # total_testset = 0
    # for i, data in tqdm(enumerate(testdataloader, 0)):
    #     points, target = data
    #     target = target[:, 0]
    #     points = points.transpose(2, 1)
    #     points, target = points.cuda(), target.cuda()
    #     classifier = classifier.eval()
    #     pred, _, _ = classifier(points)
    #     pred_choice = pred.data.max(1)[1]
    #     correct = pred_choice.eq(target.data).cpu().sum()
    #     total_correct += correct.item()
    #     total_testset += points.size()[0]
    #
    # print("final accuracy {}".format(total_correct / float(total_testset)))


if __name__ == "__main__":
    main()
