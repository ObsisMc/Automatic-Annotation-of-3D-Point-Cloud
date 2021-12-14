from visualize.TrackingV.extract_partial_point_clouds_from_kitti import main as extractTra
from visualize.TrackingV.visualize import main as VTra


def main():
    extractTra(True, sceneid='0003')
    # VTra(True, frame=0, sceneid='0003', category='Car',pid=2)


if __name__=="__main__":
    main()
