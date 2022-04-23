import os
import torch
import common_utils.cfgs as Config

config_common = Config.load_train_common()


def getDataSetPath():
    return config_common["dataset_path"]


def checkCheckpointFile():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")


def saveCheckPoint(net, epoch, loss):
    model = net.__class__.__name__

    filefolder = os.path.join(config_common["ckpt_out_path"], model)
    if not os.path.exists(filefolder):
        os.makedirs(filefolder)

    torch.save(net.state_dict(), os.path.join(filefolder, 'ckpt_epc%d_%f.pth' % (epoch, loss)))
