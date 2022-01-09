from visualize.TrackingV.extract_partial_point_clouds_from_kitti import main as extractTra
from visualize.TrackingV.visualize import main as VTra
from visualize.TrackingV.visualizeTraining import main as Vtraining
from visualize.TrackingV.extract_trainingpoint import main as extractPoint

prefix = "../Data/"


def main():
    # extract 和 V 最好分开运行
    # extractTra(True, sceneid='0000', dilate=1.5,generror=True,label=prefix+"kitti/training/tracking/label_02_training/")
    # extractPoint(True, sceneid='0000', dilate=1.5, generror=True,
    #              label=prefix + "kitti/training/tracking/label_02_training/")
    # VTra(True, sceneid='0000', category='Van', frame=51, pid=0, subcate=0)
    Vtraining("../Data/Mydataset/0000/groundtruth/Van_0/bbox0.npy", idx=23, gt=0)


if __name__ == "__main__":
    main()
