import os
import sys
import yaml
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))


def load_visual(choice: str) -> dict:
    path = os.path.join(root, "visual_modul/visual_cfg.yaml")
    return yaml.load(open(path, encoding="utf-8"), Loader=yaml.FullLoader)[choice]


def load_model_visual():
    path = os.path.join(root, "visual_modul/model_visual.yaml")
    return yaml.load(open(path, encoding="utf-8"), Loader=yaml.FullLoader)["runs_root"]


def load_pillar_vfe():
    path = os.path.join(root, "dataset/pillar_cfg.yaml")
    data_dict = yaml.load(open(path, encoding="utf-8"), Loader=yaml.FullLoader)["data_dict"]

    # pre-process
    size = data_dict["voxel_size"] = np.array(data_dict["voxel_size"])
    vrange = data_dict["point_cloud_range"] = np.array(data_dict["point_cloud_range"])
    data_dict["image_shape"] = (vrange[3:6] - vrange[0:3]) / size
    return data_dict
