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


def load_pillar_vfe():
    path = os.path.join(root, "dataset/pillar_cfg.yaml")
    with open(path, encoding="utf-8") as f:
        pillar_cfg = yaml.load(f, Loader=yaml.FullLoader)["pillar_config"]

        # pre-process
        size = pillar_cfg["VOXEL_SIZE"] = np.array(pillar_cfg["VOXEL_SIZE"])
        vrange = pillar_cfg["POINT_CLOUD_RANGE"] = np.array(pillar_cfg["POINT_CLOUD_RANGE"])

        pillar_cfg["VOXEL_SIZE"][2] = pillar_cfg["POINT_CLOUD_RANGE"][5] - pillar_cfg["POINT_CLOUD_RANGE"][2]
        pillar_cfg["IMAGE_SIZE"] = (vrange[3:6] - vrange[0:3]) / size
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


if __name__ == "__main__":
    load_pillar_data_template(2)
