from testNet import PointNetCls as Net
from OurNet.dataset.MyDataSet import MyDataSet

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

blue = lambda x: '\033[94m' + x + '\033[0m'


def main(epochs=10, batch=1, shuffle=False, wokers=4):
    dataset = MyDataSet()
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)),
                                                                           len(dataset) - int(0.8 * len(dataset))])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)
    valid_dataset = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)

    print(blue('# of training samples: %d' % len(train_dataset)))
    print(blue('# of validation samples: %d' % len(valid_dataset)))

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    for epoch in range(epochs):
        n = 0
        for i, data in enumerate(train_dataloader):
            points, label = data[0][1], int(data[1][-1])
            print(points.shape)
            optimizer.zero_grad()
            pred = net(points)
            loss = F.cross_entropy(pred, torch.tensor(label, dtype=torch.long))
            loss.backward()
            optimizer.step()
            n += 1
            if n % 100 == 0:
                print(loss)


if __name__ == "__main__":
    main()
