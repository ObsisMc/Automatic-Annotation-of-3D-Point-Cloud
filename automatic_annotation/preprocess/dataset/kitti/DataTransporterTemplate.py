import yaml
import os


class DataTransporterTemplate:
    def __init__(self):
        with open("kitti_cfg.yaml") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.kitti_cfg = self.config["kitti"]
        self.dst_path = {"train": [], "test": []}

    def transport(self):
        return NotImplemented

    def process(self):
        return

    def check(self):
        for i, key in self.kitti_cfg:
            for path in self.kitti_cfg[key]:
                assert os.path.exists(path)
