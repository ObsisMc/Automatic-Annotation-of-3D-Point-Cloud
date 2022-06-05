from dataset.dataset_utils.Sampler import Sampler
import torch
from models.detector.SiameseNet import Siamese2c


def test(points1, points2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sampler = Sampler()
    points1, points2 = sampler.fps(points1), sampler.fps(points2)
    points1, points2 = torch.from_numpy(points1).unsqueeze(0).type(torch.float32).to(device), torch.from_numpy(points2).unsqueeze(0).type(torch.float32).to(device)
    net = Siamese2c()
    net.load_state_dict(torch.load("/home2/lie/zhangrh/InnovativePractice1_SUSTech/OurNet/checkpoints/Siamese2c/negative/ckpt_epc160_0.008512.pth", map_location='cuda:0'))
    net.to(device)
    output = net.forward(points1, points2)
    return output
