import numpy as np
import torch
import torchvision
import torch.utils.data
import torch.nn as nn
import os
import cv2

import common_utils.cfgs as Config

data_path = "/home2/lie/InnovativePractice2/data/kitti/tracking/extracted_08/"
ckpt = "/home2/lie/zhangsh/InnovativePractice1_SUSTech/OurNet/checkpoints/simpleSizeNet/0.00333880287960071.pth"


def test():
    cfgs = Config.load_train_common()
    dataset_path = cfgs["dataset_path"]
    ckpt_out_path = cfgs["ckpt_out_path"]
    train_par = cfgs["training_parameters"]
    batch, shuffle, workers = train_par["batch_size"], train_par["shuffle"], train_par["workers"]
    epochs = train_par["epochs"]
    early_stop = train_par["early_stop"]
    lr = train_par["learning_rate"]
    device = "cuda:%d" % train_par["cudaidx"] if torch.cuda.is_available() else "cpu"
    model = torchvision.models.resnet101(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    model = model.to(device)
    # load checkpoint
    if os.path.isfile(ckpt):
        print("=> loading checkpoint '{}'".format(ckpt))
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint)
        print("Successfully loaded checkpoint '{}'".format(ckpt))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt))
    # load data
    scenes = os.listdir(data_path)
    for scene in scenes:
        print("Testing scene: {}".format(scene))
        vehicles = os.listdir(data_path + scene)
        for vehicle in vehicles:
            point_clouds = os.listdir(data_path + scene + "/" + vehicle + "/points/")
            bev = np.zeros((128, 128))
            for point_cloud in point_clouds:
                cloud = np.load(data_path + scene + "/" + vehicle + "/points/" + point_cloud)
                for i in range(cloud.shape[0]):
                    x = int(cloud[i][0] / 0.05) + 64
                    y = int(cloud[i][1] / 0.05) + 64
                    if 0 <= x < 128 and 0 <= y < 128:
                        # bev[x][y] += cloud[i][2]
                        bev[x][y] += 1
            bev = bev.astype(np.uint8)
            bev = cv2.equalizeHist(bev)
            bev = (bev - np.mean(bev)) / np.std(bev)
            bev = bev.astype(np.float32)
            bev = torch.from_numpy(bev)
            bev = bev.to(device)
            bev = bev.unsqueeze(0)
            bev = bev.unsqueeze(0)
            output = model.forward(bev)
            output = output.detach().cpu().numpy()
            l = output[0][0]
            w = output[0][1]
            labels = os.listdir(data_path + scene + "/" + vehicle + "/labels/")
            for label in labels:
                # write output to file
                with open(data_path + scene + "/" + vehicle + "/labels/" + label, "r") as f:
                    old_label = f.readline().split(" ")
                    old_label[3] = str(l)
                    old_label[5] = str(w)
                with open(data_path + scene + "/" + vehicle + "/labels/" + label, "w") as f:
                    f.write(" ".join(old_label))


if __name__ == "__main__":
    test()