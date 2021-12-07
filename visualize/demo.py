from extract_partial_point_clouds_from_kitti import main as extract
from visualize import main as V


def main():
    extract(True, idx='000000', category='car')
    V(True, 0, '000000', 'car')


if __name__=="__main__":
    main()
