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


def saveNewLabel(pred, info):
    """
    used by smoothTrajTraining

    pred: [[dx, dy, dangle],...] torch.Tensor
    info: [[scene, tid, frame],...] torch.Tensor
    """
    cfg = Config.load_visual("extract_root")
    extractpath = cfg["outputroot"][cfg["keep_world_coord"]]
    change_root = os.path.join(extractpath, info["scene"][0], info["tid"][0], "labels")
    for i in range(pred.shape[0]):
        label = os.path.join(change_root, info["frame"][i][0] + ".txt")
        new_label = os.path.join(change_root, info["frame"][i][0] + "_test" + ".txt")
        with open(label, "r") as f:
            label_list = f.readline().split(" ")  # [x, y, z, l, h, w, angle]
            label_list[0] = str(float(pred[i][0]) + float(label_list[0]))
            label_list[1] = str(float(pred[i][1]) + float(label_list[1]))
            label_list[-1] = str(float(pred[i][2]) + float(label_list[-1]))
            with open(new_label, "w") as f2:
                f2.write(" ".join(label_list))
