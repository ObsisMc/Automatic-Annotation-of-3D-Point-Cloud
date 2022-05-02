import copy
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


def load_pillar_vfe(point_range=None, voxel_size=None):
    """
    point_range: [lower_x, ly, lz, upper_x, uy, uz]
    """
    path = os.path.join(root, "dataset/pillar_cfg.yaml")
    with open(path, encoding="utf-8") as f:
        pillar_cfg = yaml.load(f, Loader=yaml.FullLoader)["pillar_config"]

        # pre-process
        size = pillar_cfg["VOXEL_SIZE"] = \
            np.array(voxel_size if voxel_size is not None else pillar_cfg["VOXEL_SIZE"])
        vrange = pillar_cfg["POINT_CLOUD_RANGE"] = \
            np.array(point_range if point_range is not None else pillar_cfg["POINT_CLOUD_RANGE"])

        pillar_cfg["VOXEL_SIZE"][2] = pillar_cfg["POINT_CLOUD_RANGE"][5] - pillar_cfg["POINT_CLOUD_RANGE"][2]  # calc z
        pillar_cfg["GRID_SIZE"] = (vrange[3:6] - vrange[0:3]) / size
    return pillar_cfg


def load_pillar_data_template(num=1) -> list:
    path = os.path.join(root, "dataset/pillar_cfg.yaml")
    with open(path, encoding="utf-8") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)["data_dict"]
        data_dicts = [copy.deepcopy(data_dict) for _ in range(num)]
    return data_dicts


def load_train_common():
    path = os.path.join(root, "models/training_cfg.yaml")
    return yaml.load(open(path, encoding="utf-8"), Loader=yaml.FullLoader)["training_common"]


def load_train_pillar_cfg():
    path = os.path.join(root, "models/training_cfg.yaml")
    return yaml.load(open(path, encoding="utf-8"), Loader=yaml.FullLoader)["pillar_cfg"]


if __name__ == "__main__":
    load_pillar_data_template(2)
