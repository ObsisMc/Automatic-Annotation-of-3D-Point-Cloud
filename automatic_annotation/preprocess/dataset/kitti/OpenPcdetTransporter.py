import shutil
import os

from DataTransporterTemplate import DataTransporterTemplate


class OpenPcdetTransporter(DataTransporterTemplate):
    def __init__(self):
        super().__init__()
        self.openpcdet_cfg = self.config["OpenPcdet"]
        self.dst_path = {"train": os.path.join(self.openpcdet_cfg["base"], self.openpcdet_cfg["train"]),
                         "test": os.path.join(self.openpcdet_cfg["base"], self.openpcdet_cfg["test"])}

    def transport(self):
        print("Check directories...")
        self.check()
        print("Check successfully")
        for i, key in enumerate(self.kitti_cfg):
            for path in self.kitti_cfg[key]:
                dir = os.path.split(path)[1]
                print("Copy %s of %s..." % (dir, key))
                shutil.copytree(path, os.path.join(self.dst_path[key], dir))
        self.process()
        return 0

    def process(self):
        return

    def check(self):
        super().check()
        for i, key in enumerate(self.dst_path):
            assert os.path.exists(self.dst_path[key])


def test():
    trans = OpenPcdetTransporter()
    trans.transport()


if __name__ == "__main__":
    test()
