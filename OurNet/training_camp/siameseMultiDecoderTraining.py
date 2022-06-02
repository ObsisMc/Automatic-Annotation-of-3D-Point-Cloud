from OurNet.models.detector.SiameseNet import SiameseMultiDecoder, SiameseAttentionMulti
from OurNet.dataset.NewDataSet import NewDataSet
from OurNet.visualization.tensorboard import TensorBoardVis
from OurNet.models.model_utils import io_utils

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

blue = lambda x: '\033[94m' + x + '\033[0m'

vis = TensorBoardVis()
vis_name = ["loss_x", "loss_y", "loss_z", "loss_angel", "loss_confidence"]


def visualize(losses, step, k):
    location_loss = 0
    for i in range(k):
        if i < 3:
            location_loss += losses[i].detach()
        vis.add_scalar(vis_name[i], losses[i].detach(), step)
    vis.add_scalar("location_loss", location_loss, step)

    return torch.sum(torch.Tensor(losses), dtype=torch.float32).detach()


def main(epochs=200, batch=4, shuffle=False, wokers=4, cudan=0):
    device = "cuda:%d" % cudan if torch.cuda.is_available() else "cpu"

    dataset = NewDataSet("/home/zrh/Data/kitti/tracking/extracted_points_entend13")
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)) - 1,
                                                                           len(dataset) - int(0.8 * len(dataset)) + 1])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)
    valid_dataset = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=shuffle, num_workers=wokers)

    print(blue('# of training samples: %d' % len(train_dataset)))
    print(blue('# of validation samples: %d' % len(valid_dataset)))

    k = 5
    # net = SiameseMultiDecoder(k)
    net = SiameseAttentionMulti(k)
    net.to(device)

    lr1 = 0.01
    lr2 = 0.05
    lr3 = 0.005
    lr4 = 0.0001
    optimizers = optim.Adam([
        {'params': net.decoders[0].parameters(), "lr": lr4},
        {"params": net.decoders[1].parameters(), "lr": lr4},
        {"params": net.decoders[2].parameters(), "lr": lr3},
        {"params": net.decoders[3].parameters(), "lr": lr4},
        {"params": net.decoders[4].parameters(), "lr": lr3}],
        lr=lr1)
    scheduler = optim.lr_scheduler.StepLR(optimizers, step_size=10, gamma=0.5)

    totalstep = 0
    for epoch in range(epochs):
        n = 0
        for i, data in enumerate(train_dataloader):
            points, label = data
            source = points[0].type(torch.FloatTensor)  # 一维卷积是在最后维度上扫的, change value in other dimension
            target = points[1].type(torch.FloatTensor)

            source, target, label = source.to(device), target.to(device), label.to(device)
            # for i in range(k):
            #     optimizers[i].zero_grad()
            optimizers.zero_grad()
            net = net.train()
            pred = net(source, target)

            loss_confidence = F.binary_cross_entropy_with_logits(pred[4][:, 0], label[:, 4])
            losses = [F.smooth_l1_loss(pred[i][:, 0], label[:, i]) for i in
                      range(0, k - 1)]  # maybe loss for angle can change to another one
            losses.append(loss_confidence)

            for j in range(k):
                losses[j].backward(retain_graph=True if j < k - 1 else False)

            optimizers.step()
            total_loss = visualize(losses, step=totalstep, k=k)
            totalstep += 1
            n += 1
            if n % 100 == 0:
                print("Epoch %d. total loss is %f: x->%f, y->%f, z->%f, angel->%f, confidence->%f " %
                      (epoch, total_loss, losses[0].detach(), losses[1].detach(), losses[2].detach()
                       , losses[3].detach(), losses[4].detach()))
                pass
        vis.add_scalar("total_loss", total_loss, epoch)
        scheduler.step()

        if epoch % 10 == 0 and epoch > 0:
            io_utils.saveCheckPoint(net, epoch, total_loss)


if __name__ == "__main__":
    main()
