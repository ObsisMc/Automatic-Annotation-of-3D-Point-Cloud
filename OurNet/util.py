import numpy as np
import os
import shutil


def normPointNum(path, padding):
    try:
        points = np.load(path)
        num = points.shape[0]
        if num >= padding:
            points = points[np.random.choice(len(points), size=padding, replace=False)]
        else:
            points = np.r_[
                points,
                points[np.random.choice(len(points), size=-num + padding, replace=True)]
            ]
    except:
        # error数据可能里面没有点所以没有生成点云文件
        points = np.array([[0, 0, 0] for _ in range(padding)])
        print(path, "has no points.")

    # print(points.shape)
    return points


def createTrainingSet(sceneid=0, maxgap=1, padding=500, id="Van_0"):
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

        # 只适用于连续帧有标签
        for gap in range(1, maxgap + 1):
            for i in range(0, len(labels) - gap):
                idx = gap + i
                label = labels[idx].rstrip("\n")
                framei = int(label.split(" ")[0])
                with open(outlabel + "{:04}.txt".format(setlen), "w") as lb:
                    lb.write(label)

                points_dir = outvelodyne + "{:04}/".format(setlen)
                if not os.path.exists(points_dir):
                    os.makedirs(points_dir)

                errorname = "point{}.npy".format(framei)
                gtname = "point{}.npy".format(framei - gap)

                # 处理点数量
                errorpoints = normPointNum(error_dir + errorname, padding)
                gtpoints = normPointNum(gt_dir + gtname, padding)

                np.save(points_dir + errorname, errorpoints)
                np.save(points_dir + gtname, gtpoints)

                setlen += 1


def checkCode(padding=800):
    path = "../Data/Mydataset/training/velodyne/"
    datap = os.listdir(path)

    nonpoint = np.array([[0, 0, 0] for _ in range(padding)])
    errorp = set()
    for i in range(len(datap)):
        points = path + datap[i]
        pts = os.listdir(points)
        for j in range(2):
            if np.all(nonpoint == np.load(os.path.join(points, pts[j]))):
                errorp.add(os.path.join(points, pts[j]))
    print(errorp)


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
    createTrainingSet(maxgap=20, padding=800)
    # checkCode(800)
    # print(checkSamePoint("../Data/Mydataset/training/velodyne/0156/point2.npy",
    #                      "../Data/Mydataset/0000/groundtruth/Van_0/point2.npy"))
