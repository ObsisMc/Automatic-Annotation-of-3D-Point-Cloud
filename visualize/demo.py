from visualize.TrackingV.extract_partial_point_clouds_from_kitti import main as extractTra
from visualize.TrackingV.visualize import main as VTra


def main():
    # extractTra(True, idx='0003', category='car')
    VTra(True, frame=0, idx='0003', category='Car',pid=0)


if __name__=="__main__":
    main()
