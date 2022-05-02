import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

import common_utils.cfgs as Config
from OurNet.dataset.PillarDataSet import SmPillarDataSet
from OurNet.models.detector.SmPillarNet import SmPillarNet


def main():
    cfgs = Config.load_train_common()
    dataset_path = cfgs["dataset_path"]
    ckpt_out_path = cfgs["ckpt_out_path"]
    train_par = cfgs["training_parameters"]
    batch, shuffle, workers = train_par["batch_size"], train_par["shuffle"], train_par["workers"]
    epochs = train_par["epochs"]

    # cuda or cpu
    device = "cuda:%d" % train_par["cudaidx"] if torch.cuda.is_available() else "cpu"

    # dataset
    dataset = SmPillarDataSet(dataset_path)
    train_num = int(len(dataset) * train_par["train_ratio"])
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_num, len(dataset) - train_num])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=shuffle,
                                                   num_workers=workers)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=shuffle,
                                                   num_workers=workers)

    # net
    net = SmPillarNet(device).to(device)
    net.train()

    # train
    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader):
            source_dict, target_dict = data
            print(source_dict["points"].shape)
            source_bev = net(source_dict, target_dict)
            print(torch.max(source_bev, dim=1)[0])
            print(source_bev.shape)
            print("stop")
    pass


if __name__ == "__main__":
    main()
