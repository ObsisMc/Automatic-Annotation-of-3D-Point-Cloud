from visualize.TrackingV.extract_partial_point_clouds_from_kitti import main as extractTra
from visualize.TrackingV.visualize import main as VTra


def main():
    # extract 和 V 最好分开运行
    extractTra(True, sceneid='0000', dilate=1.5,generror=True)
    # VTra(True, sceneid='0000', category='Car', frame=117,pid=5,subcate=1)


if __name__ == "__main__":
    main()
