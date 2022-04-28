import os
import numpy


class DataSetCreator:
    def __init__(self, datapath):
        self.datapath = datapath

    def gapTwoInTradj(self, gap=1, scenen=1, starts=0, type=("Car")) -> list:
        """
        data_root
          |-> 0000 // scene
                |-> Car#1 // type and trajectoryId
                      |-> points
                        |-> 000000.npy // points cloud in frameId
                      |-> labels
                        |-> 000000.txt
        @return
            [[source_path, target_path],[],...]
        """
        dataset = []

        scene_list = sorted(os.listdir(self.datapath), key=lambda x: int(x))[starts:starts + scenen]
        for scene in scene_list:
            tid_dir = os.path.join(self.datapath, scene)
            tid_list = os.listdir(tid_dir)
            for tid in tid_list:
                if tid.split("#")[0] not in type:
                    continue
                object_points_dir = os.path.join(tid_dir, tid, "points")
                points_list = sorted(os.listdir(os.path.join(object_points_dir)), key=lambda x: int(x.rstrip(".npy")))
                # labels_list = sorted(os.listdir(os.path.join(object_list, "labels")), key=lambda x: int(x.rstrip(".txt")))

                datan = len(points_list) - gap
                for i in range(gap, datan):
                    points = [os.path.join(object_points_dir, points_list[i - gap]),
                              os.path.join(object_points_dir, points_list[i])]  # source, target
                    # label = labels_list[i]  # target label
                    dataset.append(points)
        return dataset
