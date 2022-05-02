import torch
import torch.utils.data
import torch.optim as optim
import common_utils.cfgs as Config
from OurNet.visualization.tensorboard import TensorBoardVis as vis

from OurNet.dataset.SmoothTrajDataSet import SmoothTrajDataSet
from OurNet.models.detector.SmoothTrajNet import SmoothTrajNet

blue = lambda x: '\033[94m' + x + '\033[0m'
def main():
    train_cfg = Config.load_train_common()
    train_para = train_cfg["training_parameters"]
    batchs, workers, shuffle = train_para["batch_size"], train_para["workers"], train_para["shuffle"]
    epochs = train_para["epochs"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    max_traj_n = 10
    dataset = SmoothTrajDataSet(train_cfg["dataset_path"], max_traj_n=max_traj_n)
    train_num = int(len(dataset) * train_para["train_ratio"])
    train_data, valid_data = torch.utils.data.random_split(dataset, [train_num, len(dataset) - train_num])
    train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=workers, batch_size=batchs, shuffle=shuffle)
    vaild_dataloader = torch.utils.data.DataLoader(train_data, num_workers=workers, batch_size=batchs, shuffle=shuffle)

    net = SmoothTrajNet(device, N=max_traj_n).to(device)

    print(blue('# of training samples: %d' % len(train_data)))
    print(blue('# of validation samples: %d' % len(valid_data)))

    n =0
    for epoch in range(epochs):
        for i, data in enumerate(train_dataloader):
            point_dicts, poses, labels = data
            poses = poses.to(device)
            pred = net(point_dicts, poses)
            # print(labels)
            n+=1
            if n%100 == 0:
                print(n)
        break


if __name__ == "__main__":
    main()
