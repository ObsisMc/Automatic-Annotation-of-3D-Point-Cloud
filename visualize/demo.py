from visualize.TrackingV.extract_partial_point_clouds_from_kitti import main as extractTra
from visualize.TrackingV.visualize import main as VTra
from visualize.TrackingV.extract_trainingpoint import main as extractPoint

prefix = "../Data/"


def main():
    # extract 和 V 最好分开运行
    # extractTra(True, sceneid='0000', dilate=1.5,generror=True,label=prefix+"kitti/training/tracking/label_02_training/")
    # extractPoint(True, sceneid='0000', dilate=1.5, generror=True,
    #              label=prefix + "kitti/training/tracking/label_02_training/")
    VTra(True, sceneid='0000', category='Cyclist', frame=0, pid=1, subcate=0)


if __name__ == "__main__":
    main()
