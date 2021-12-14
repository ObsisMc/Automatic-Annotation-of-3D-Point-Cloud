from visualize.DetectV.extract_partial_point_clouds_from_kitti import main as extractObj
from visualize.DetectV.visualize import main as VObj
from visualize.TrackingV.extract_partial_point_clouds_from_kitti import main as extractTra
from visualize.TrackingV.visualize import main as VTra


def main():
    extractTra(True, idx='000000', category='car')
    VTra(True, 0, '000000', 'car')


if __name__=="__main__":
    main()
