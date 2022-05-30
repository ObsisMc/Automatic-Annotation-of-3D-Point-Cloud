import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointNet(nn.Module):
    def __init__(self, k=3):
        super(SimplePointNet, self).__init__()
        self.backbone = self.getBackone(k=k)

    def getBackone(self, k=3, mlp=[64, 128, 512, 1024]):
        seq = nn.Sequential()
        input_d = k
        for i, output_d in enumerate(mlp):
            seq.add_module("conv%d" % (i + 1), nn.Conv2d(input_d, output_d, (1, 1)))
            seq.add_module("bn%d" % (i + 1), nn.BatchNorm2d(output_d))
            seq.add_module("relu%d" % (i + 1), nn.LeakyReLU())
            input_d = output_d
        return seq

    def forward(self, x):
        """
        x: (B,3,N,C)
        """
        x = self.backbone(x)
        x = torch.max(x, 2, keepdim=False)[0]
        x = x.view(-1, 1024 * x.shape[2])
        return x


class Siamese2c(nn.Module):
    def __init__(self, k=5):
        super(Siamese2c, self).__init__()
        self.pointfeat = SimplePointNet()
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
        x = torch.cat((x1.unsqueeze(3), x2.unsqueeze(3)), 3).permute(0, 2, 1, 3)
        x = self.pointfeat(x)
        x = F.leaky_relu(self.dropout(self.fc1(x)))
        x = F.leaky_relu(self.dropout(self.fc2(x)))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        return x


class SiameseMultiDecoder(nn.Module):
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2048, 512)
            self.fc2 = nn.Linear(512, 64)
            self.fc3 = nn.Linear(64, 1)

            self.bn1 = nn.BatchNorm1d(1024)
            self.bn2 = nn.BatchNorm1d(512)
            self.bn3 = nn.BatchNorm1d(64)


        def forward(self, x):
            x = F.leaky_relu(self.bn2(self.fc1(x)))
            x = F.leaky_relu(self.bn3(self.fc2(x)))
            x = self.fc3(x)
            return x

    def __init__(self, k=5):
        super().__init__()
        self.pointfeat = SimplePointNet()
        self.k = k
        self.decoders = [self.Decoder().to("cuda:0") for _ in range(k)]  # x,y,z,angel,confidence

    def forward(self, x1, x2):
        """
        x1 and x2: (B,N,3)
        """
        x = torch.cat((x1.unsqueeze(3), x2.unsqueeze(3)), 3).permute(0, 2, 1, 3)
        x = self.pointfeat(x)

        y = [self.decoders[i](x) for i in range(self.k)]
        return y
