import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
import torch.utils.data
import os
import common_utils.cfgs as Config
from OurNet.dataset.SimpleSizeDataSet import SimpleSizeDataSet


def main():
    cfgs = Config.load_train_common()
    dataset_path = cfgs["dataset_path"]
    ckpt_out_path = cfgs["ckpt_out_path"]
    train_par = cfgs["training_parameters"]
    batch, shuffle, workers = train_par["batch_size"], train_par["shuffle"], train_par["workers"]
    epochs = train_par["epochs"]
    early_stop = train_par["early_stop"]
    lr = train_par["learning_rate"]
    device = "cuda:%d" % train_par["cudaidx"] if torch.cuda.is_available() else "cpu"

    dataset = SimpleSizeDataSet(dataset_path)
    print("dataset size:", len(dataset))
    train_num = int(len(dataset) * train_par["train_ratio"])
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_num, len(dataset) - train_num])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=shuffle,
                                               num_workers=workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=shuffle,
                                               num_workers=workers)
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
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    min_loss = 10000
    count = 0
    print("start training")
    for epoch in range(10000):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # convert to float tensor
            inputs = inputs.float()
            labels = labels.float()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            inputs = torch.unsqueeze(inputs, dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 10 == 0:
            # validation
            val_loss = 0.0
            l_error = 0.0
            w_error = 0.0
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs = inputs.float()
                    labels = labels.float()
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs = torch.unsqueeze(inputs, dim=1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    l_error += torch.mean(torch.abs(outputs[:, 0] - labels[:, 0])).item()
                    w_error += torch.mean(torch.abs(outputs[:, 1] - labels[:, 1])).item()
            running_loss /= len(train_loader)
            val_loss /= len(valid_loader)
            l_error /= len(valid_loader)
            w_error /= len(valid_loader)
            if val_loss < min_loss:
                min_loss = val_loss
                count = 0
                pth_name = ckpt_out_path + str(min_loss) + ".pth"
                torch.save(model.state_dict(), pth_name)
            else:
                count += 1
                if count > 5 and early_stop:
                    print("Early stopping")
                    break
            print(
                'Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Length Average Error: {:.4f}, Width Average Error: {:.4f}'.format(
                    epoch, running_loss, val_loss, l_error, w_error))


if __name__ == '__main__':
    main()

