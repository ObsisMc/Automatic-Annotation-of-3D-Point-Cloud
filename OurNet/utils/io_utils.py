import os

import sys
import torch

from OurNet.models.SiameseNet import Siamese2c
from OurNet.models.NetPractice1 import PointNetCls, PointNetPred
from OurNet.models.PointNet import SimplePointNet


def checkCheckpointFile():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")


def saveCheckPoint(net, epoch, loss):
    model = "unknown"
    if isinstance(net, Siamese2c):
        model = "Siamese2c"
    elif isinstance(net, PointNetCls):
        model = "PointNetCls"
    elif isinstance(net, PointNetPred):
        model = "PointNetPred"
    elif isinstance(net, SimplePointNet):
        model = "SimplePointNet"

    filefolder = os.path.join("checkpoints", model)
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)

    torch.save(net.state_dict(), 'ckpt_epc%d_%f.pth' % (epoch, loss))
