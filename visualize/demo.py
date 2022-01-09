from visualize.TrackingV.visualizeTraining import main as Vtraining

prefix = "../Data/"


def main():
    # extract 和 V 最好分开运行
    # extractTra(True, sceneid='0000', dilate=1.5,generror=True,label=prefix+"kitti/training/tracking/label_02_training/")
    # extractPoint(True, sceneid='0000', dilate=1.5, generror=True,
    #              label=prefix + "kitti/training/tracking/label_02_training/")
    # VTra(True, sceneid='0000', category='Van', frame=0, pid=0, subcate=0)
    Vtraining("../Data/Mydataset/0000/groundtruth/Van_0/bbox0.npy", idx=1, gt=0)


if __name__ == "__main__":
    main()
