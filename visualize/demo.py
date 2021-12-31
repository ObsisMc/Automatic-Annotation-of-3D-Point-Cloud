from visualize.TrackingV.extract_partial_point_clouds_from_kitti import main as extractTra
from visualize.TrackingV.visualize import main as VTra

prefix = "../Data/"
def main():
    # extract 和 V 最好分开运行
    extractTra(True, sceneid='0000', dilate=1.5,generror=True,label=prefix+"kitti/training/tracking/label_02_training/")
    # VTra(True, sceneid='0000', category='Pedestrian', frame=3, pid=2, subcate=1)


if __name__ == "__main__":
    main()
