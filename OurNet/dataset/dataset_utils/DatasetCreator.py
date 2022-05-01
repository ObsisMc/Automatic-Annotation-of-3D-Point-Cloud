import copy
import os
import numpy as np


def getLW(label_path):
    with open(label_path) as f:
        label = f.readline()
    label = label.split(' ')
    return np.array([float(label[3]), float(label[5])])


class DataSetCreator:
    def __init__(self, datapath):
        self.datapath = datapath

    def gapTwoInTradj(self, gap=1, scenen=10, starts=0, type=("Car", "Van")) -> list:
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

    def getWorldTrajectary(self, starts=0, scene_n=10, type=("Car"), max_traj_n=10, cache=False):
        """
        get series boxes and points of a object's trajectory
        Pay attention: some trajectory isn't continuous
        """
        dataset = []

        scene_list = sorted(os.listdir(self.datapath), key=lambda x: int(x))[starts:starts + scene_n]
        for scene in scene_list:
            tid_dir = os.path.join(self.datapath, scene)
            tid_list = os.listdir(tid_dir)
            for tid in tid_list:
                if tid.split("#")[0] not in type:
                    continue
                object_labels_dir = os.path.join(tid_dir, tid, "labels")
                object_points_dir = os.path.join(tid_dir, tid, "points")
                labels_list = sorted(os.listdir(object_labels_dir), key=lambda x: int(x.rstrip(".txt")))
                points_list = sorted(os.listdir(object_points_dir), key=lambda x: int(x.rstrip(".npy")))

                window = [[], []]  # [[points], [labels]]
                for i in range(len(labels_list)):
                    if len(window[0]) < max_traj_n:
                        window[0].append(os.path.join(object_points_dir, points_list[i]))
                        window[1].append(os.path.join(object_labels_dir, labels_list[i]))
                    elif len(window[0]) == max_traj_n:
                        dataset.append(copy.deepcopy(window))
                        window[0].pop(0)
                        window[1].pop(0)
                if len(window[0]) > 0:
                    dataset.append(window)
        return dataset

    def severalFrameInTraj(self, length=5, scenen=21, starts=0, type=("Car", "Pedestrian", "Cyclist")):
        pointSet = []
        labelSet = []
        scene_list = sorted(os.listdir(self.datapath), key=lambda x: int(x))[starts:starts + scenen]
        for scene in scene_list:
            tid_dir = os.path.join(self.datapath, scene)
            tid_list = os.listdir(tid_dir)
            for tid in tid_list:
                if tid.split("#")[0] not in type:
                    continue
                object_points_dir = os.path.join(tid_dir, tid, "points")
                object_labels_dir = os.path.join(tid_dir, tid, "labels")
                points_list = sorted(os.listdir(os.path.join(object_points_dir)), key=lambda x: int(x.rstrip(".npy")))
                labels_list = sorted(os.listdir(os.path.join(object_labels_dir)), key=lambda x: int(x.rstrip(".txt")))
                num_frame = len(points_list)
                if num_frame <= length:
                    # points is all the trajectory
                    points = [os.path.join(object_points_dir, points_list[i]) for i in range(num_frame)]
                    label = getLW(os.path.join(object_labels_dir, labels_list[0]))
                    pointSet.append(points)
                    labelSet.append(label)
                else:
                    for i in range(num_frame - length + 1):
                        points = [os.path.join(object_points_dir, points_list[i + j]) for j in range(length)]
                        label = getLW(os.path.join(object_labels_dir, labels_list[0]))
                        pointSet.append(points)
                        labelSet.append(label)
                    # generate a piece of data by all frame
                    points = [os.path.join(object_points_dir, points_list[i]) for i in range(num_frame)]
                    label = getLW(os.path.join(object_labels_dir, labels_list[0]))
                    pointSet.append(points)
                    labelSet.append(label)
        return pointSet, labelSet
