import os
import torch


def checkCheckpointFile():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")


def saveCheckPoint(net, epoch, loss):
    model = net.__class__.__name__

    filefolder = os.path.join("checkpoints", model)
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)

    torch.save(net.state_dict(), 'ckpt_epc%d_%f.pth' % (epoch, loss))
