from visualize.TrackingV.extract_partial_point_clouds_from_kitti import main as extractTra
from visualize.TrackingV.visualize import main as VTra


def main():
    # extractTra(True, sceneid='0003')
    VTra(True, sceneid='0003', category='Car', frame=50,pid=0)


if __name__=="__main__":
    main()
