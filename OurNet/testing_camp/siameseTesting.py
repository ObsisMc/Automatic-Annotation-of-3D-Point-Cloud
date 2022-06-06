import torch
import torch.nn
import torch.utils.data
import torch.nn.functional as F
import os
import numpy as np

from OurNet.models.detector.SiameseNet import Siamese2c, SiameseAttentionMulti, SiameseMultiDecoder
from OurNet.dataset.NewDataSet import NewDataSet
from OurNet.evaluation.eval_utils import EvalLog

from tqdm import tqdm

blue = lambda x: '\033[94m' + x + '\033[0m'


def eval(batchs=1, workers=4, shuffle=False, find_best=False, mode=0, ratio=0.05):
    def init_dataset_net(model: int, device: str):
        dataset_tmp = net_tmp = None
        ckpt_root = "/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech/OurNet/checkpoints/"
        dataset_root = "/home/zrh/Data/kitti/tracking/extracted_points_canonical/"
        ckp_pth = []
        if model == 0:  # Siamese2C
            ckpt_root = os.path.join(ckpt_root, "Siamese2c/test")
            if not find_best:
                ckp_pth = [os.path.join(ckpt_root, "ckpt_epc140_0.009079.pth")]
            else:
                ckp_pth = [os.path.join(ckpt_root, ckpt) for ckpt in os.listdir(ckpt_root)]
            dataset_tmp = NewDataSet(dataset_root)
            net_tmp = Siamese2c().to(device)
        elif model == 1:  # SiameseMultiDecoder
            ckpt_root = os.path.join(ckpt_root, "SiameseMultiDecoder/test/")
            if not find_best:
                ckp_pth = [os.path.join(ckpt_root, "ckpt_epc80_0.002702.pth")]
            else:
                ckp_pth = [os.path.join(ckpt_root, ckpt) for ckpt in os.listdir(ckpt_root)]
            dataset_tmp = NewDataSet(dataset_root)
            net_tmp = SiameseMultiDecoder().to(device)
        elif model == 2:  # SiameseAttentionMulti
            ckpt_root = os.path.join(ckpt_root, "SiameseAttentionMulti/test/")
            if not find_best:
                ckp_pth = [os.path.join(ckpt_root, "ckpt_epc30_0.035830.pth")]
            else:
                ckp_pth = [os.path.join(ckpt_root, ckpt) for ckpt in os.listdir(ckpt_root)]
            dataset_tmp = NewDataSet(dataset_root)
            net_tmp = SiameseAttentionMulti().to(device)
        assert dataset_tmp and net_tmp and ckp_pth

        return dataset_tmp, net_tmp, ckp_pth

    def calc_error(lb: torch.Tensor, pd: torch.Tensor):
        loss_x = torch.sum(torch.abs(lb[0] - pd[0])).detach()
        loss_y = torch.sum(torch.abs(lb[1] - pd[1])).detach()
        loss_z = torch.sum(torch.abs(lb[2] - pd[2])).detach()
        loss_angel = torch.sum(torch.abs(lb[3] - pd[3])).detach()

        loss_confidence = F.binary_cross_entropy_with_logits(pd[4], lb[4]).detach()

        return loss_x, loss_y, loss_z, loss_angel, loss_confidence

    def confusion_metric(tpn, tnn, fpn, fnn, alpha=1):
        p_accu = tpn / (tpn + fpn)
        p_recall = tpn / (tpn + fnn)
        n_accu = tnn / (tnn + fnn)
        n_recall = tnn / (tnn + fpn)
        p_f1 = (1 + alpha ** 2) * p_accu * p_recall / (alpha ** 2 * p_accu + p_recall)
        n_f1 = (1 + alpha ** 2) * n_accu * n_recall / (alpha ** 2 * n_accu + n_recall)
        return [p_accu, p_recall, p_f1, n_accu, n_recall, n_f1]

    batchs, workers, shuffle = batchs, workers, shuffle
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dataset, net, ckpt_paths = init_dataset_net(model=mode, device=device)
    valid_len = int(len(dataset) * ratio)
    train_len = len(dataset) - valid_len
    _, valid_data = torch.utils.data.random_split(dataset, [train_len, valid_len])
    vaild_dataloader = torch.utils.data.DataLoader(valid_data, num_workers=workers, batch_size=batchs, shuffle=shuffle)

    print(blue('# of validation samples: %d' % valid_len))

    pbar = tqdm(total=valid_len)
    pbar.set_description("Eval process")

    error_x = error_y = error_z = error_angel = 0
    fp = fn = tp = tn = 0

    logger = EvalLog("/home/zrh/Repository/gitrepo/InnovativePractice1_SUSTech/OurNet/evaluation", net)

    best_adjust_ckpt = None
    best_adjust = 1e10
    best_confidence_ckpt = None
    best_confidence = 0
    for i, ckpt_path in enumerate(ckpt_paths):
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        pbar.reset(valid_len)
        for i, data in enumerate(vaild_dataloader):
            net.eval()
            points, label = data
            source = points[0].type(torch.FloatTensor)
            target = points[1].type(torch.FloatTensor)
            source, target, label = source.to(device), target.to(device), label.to(device)

            pred = net(source, target)
            pred = pred.squeeze()
            label = label.squeeze()

            loss_x, loss_y, loss_z, loss_angel, loss_confidence = calc_error(label, pred)
            exist = 1 if torch.sigmoid(loss_confidence) > 0.5 else 0

            sample_tp = (label[4].detach() == 1 and exist == 1)
            tn += 1 if label[4].detach() == 0 and exist == 0 else 0
            tp += 1 if sample_tp else 0
            fn += 1 if label[4].detach() == 1 and exist == 0 else 0
            fp += 1 if label[4].detach() == 0 and exist == 1 else 0

            error_x += loss_x.detach() * sample_tp
            error_y += loss_y.detach() * sample_tp
            error_z += loss_z.detach() * sample_tp
            error_angel += loss_angel.detach() * sample_tp
            pbar.update(1)

        avg_e_x = error_x / tp
        avg_e_y = error_y / tp
        avg_e_z = error_z / tp
        avg_e_angel = error_angel / tp

        # log
        p_accu, p_recall, p_f1, n_accu, n_recall, n_f1 = confusion_metric(tp, tn, fp, fn)
        ckpt_name = os.path.split(ckpt_path)[1]
        prompt = "-> %s:\n " \
                 "Pose: error_x: %f, error_y: %f, error_z: %f, error_angel: %f\n " \
                 "Positive Sample: accu: %f, recall: %f, f1: %f\n " \
                 "Negative Sample: accu: %f, recall: %f, f1: %f\n" % \
                 (ckpt_name, avg_e_x, avg_e_y, avg_e_z, avg_e_angel,
                  p_accu, p_recall, p_f1, n_accu, n_recall, n_f1)
        logger.log_eval_ckpt(prompt)
        print("\n" + prompt)

        # find best
        total_pose_loss = 2 * (avg_e_x + avg_e_y + avg_e_z) + avg_e_angel
        total_f1 = p_f1 + n_f1
        if best_confidence < total_f1:
            best_confidence_ckpt = ckpt_name
            best_confidence = total_f1
        if best_adjust > total_pose_loss:
            best_adjust = total_pose_loss
            best_adjust_ckpt = ckpt_name
    best_prompt = "\nBest:\nPose adjustment-> name: %s, total loss: %f\n" \
                  "Confidence-> name: %s, total_f1: %f\n" % \
                  (best_adjust_ckpt, best_adjust, best_confidence_ckpt, best_confidence)
    logger.log_eval_ckpt(best_prompt)
    print(best_prompt)


if __name__ == "__main__":
    eval(find_best=True)
