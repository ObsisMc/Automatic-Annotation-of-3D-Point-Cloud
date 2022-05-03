import torch
import torch.nn as nn
import torch.nn.functional as F


class NewPointNet(nn.Module):
    def __init__(self, k=3):
        super(NewPointNet, self).__init__()
        self.backbone = self.getBackone(k=k)

    def getBackone(self, k=3, mlp=[64, 128, 512, 1024]):
        seq = nn.Sequential()
        input_d = k
        for i, output_d in enumerate(mlp):
            print(i, output_d)
            seq.add_module("conv%d" % (i + 1), nn.Conv1d(input_d, output_d, 1))
            seq.add_module("bn%d" % (i + 1), nn.BatchNorm1d(output_d))
            seq.add_module("relu%d" % (i + 1), nn.LeakyReLU())
            input_d = output_d
        return seq

    def forward(self, x):
        x = self.backbone(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024 * x.shape[2])
        return x


class NewModel(nn.Module):
    def __init__(self, k=5):
        super(NewModel, self).__init__()
        self.pointfeat = NewPointNet()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, k)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x1, x2):
        """
        x1 and x2: (B,N,3)
        """
        x1 = self.pointfeat(x1.permute(0, 2, 1))
        x2 = self.pointfeat(x2.permute(0, 2, 1))
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.dropout(self.fc1(x)))
        x = F.leaky_relu(self.dropout(self.fc2(x)))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        return x