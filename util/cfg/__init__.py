import os
import sys
import yaml

root = os.path.dirname(os.path.abspath(__file__))


def load_visual(choice: str) -> dict:
    path = os.path.join(root, "visual_modul/visual_cfg.yaml")
    return yaml.load(open(path, encoding="utf-8"), Loader=yaml.FullLoader)[choice]
