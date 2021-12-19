import numpy as np
import os
import shutil


def createTrainingSet(sceneid=0, id="Van_0"):
    error_dir = "../Data/Mydataset/{:04}/error/{}/".format(sceneid, id)
    gt_dir = "../Data/Mydataset/{:04}/groundtruth/{}/".format(sceneid, id)
    label_dir = "../Data/Mydataset/{:04}/label/{}.txt".format(sceneid, id)

    outlabel = "../Data/Mydataset/training/label/"
    outvelodyne = "../Data/Mydataset/training/velodyne/"

    if not os.path.exists(outlabel):
        os.makedirs(outlabel)
    if not os.path.exists(outvelodyne):
        os.makedirs(outvelodyne)

    with open(label_dir, "r") as f:
        labels = f.readlines()
        setlen = 1
        for i in range(1, len(labels)):
            label = labels[i].rstrip("\n")
            frame = int(label[0])
            with open(outlabel + "{:04}.txt".format(setlen), "w") as lb:
                lb.write(label)

            points_dir = outvelodyne + "{:04}/".format(setlen)
            if not os.path.exists(points_dir):
                os.makedirs(points_dir)

            errorname = "point{}.npy".format(frame)
            gtname = "point{}.npy".format(frame - 1)
            shutil.copy(error_dir + errorname, points_dir + errorname)
            shutil.copy(gt_dir + gtname, points_dir + gtname)
            setlen += 1


def checkSamePoint(p1_path, p2_path):
    """

    Args: they are all .npy
        p1_path:
        p2_path:

    Returns: boolean
    """
    points1 = np.load(p1_path)
    points2 = np.load(p2_path)
    return np.all(points2 == points1)


if __name__ == "__main__":
    # createTrainingSet()
    print(checkSamePoint("../Data/Mydataset/training/velodyne/0002/point1.npy",
                         "../Data/Mydataset/0000/groundtruth/Van_0/point1.npy"))
