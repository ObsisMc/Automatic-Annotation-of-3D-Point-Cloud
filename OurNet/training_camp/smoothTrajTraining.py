import torch
import torch.utils.data
import torch.optim as optim
import common_utils.cfgs as Config

from OurNet.dataset.SmoothTrajDataSet import SmoothTrajDataSet
from OurNet.models.detector.SmoothTrajNet import SmoothTrajNet


def main():
    train_cfg = Config.load_train_common()
    train_para = train_cfg["training_parameters"]
    batchs, workers, shuffle = train_para["batch_size"], train_para["workers"], train_para["shuffle"]
    epochs = train_para["epochs"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset = SmoothTrajDataSet(train_cfg["dataset_path"])
    train_num = int(len(dataset) * train_para["train_ratio"])
    train_data, valid_data = torch.utils.data.random_split(dataset, [train_num, len(dataset) - train_num])
    train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=workers, batch_size=batchs, shuffle=shuffle)
    vaild_dataloader = torch.utils.data.DataLoader(train_data, num_workers=workers, batch_size=batchs, shuffle=shuffle)

    net = SmoothTrajNet(device).to(device)
    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader):
            point_dicts, poses, labels = data
            net(point_dicts, poses)
            break
        break


if __name__ == "__main__":
    main()
