import argparse
import numpy as np
import utils
from calibration import Calibration

output = "output"


class pseduoargs():
    def __init__(self, idx, cate, oriangle):
        self.idx = idx
        self.category = cate
        self.oriangle = oriangle


def main(shrun=False, idx='000936', category='car', oriangle=False):
    if not shrun:
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--idx', type=str, default='000936',
                            help='specify data index: {idx}.bin')
        parser.add_argument("--oriangle", action="store_true")
        parser.add_argument('--category', type=str, default='car',
                            help='specify the category to be extracted,' +
                                 '{ \
                                     Car, \
                                     Van, \
                                     Truck, \
                                     Pedestrian, \
                                     Person_sitting, \
                                     Cyclist, \
                                     Tram \
                                 }')
        args = parser.parse_args()
    else:
        args = pseduoargs(idx, category, oriangle)

    points_path = 'kitti/training/velodyne/{}.bin'.format(args.idx)
    label_path = 'kitti/training/label_2/{}.txt'.format(args.idx)
    calib_path = 'kitti/training/calib/{}.txt'.format(args.idx)

    calib = Calibration(calib_path)
    points = utils.load_point_clouds(points_path)
    bboxes = utils.load_3d_boxes(label_path, args.category)
    bboxes = calib.bbox_rect_to_lidar(bboxes)

    corners3d = utils.boxes_to_corners_3d(bboxes)
    points_flag = utils.is_within_3d_box(points, corners3d)

    points_is_within_3d_box = []
    for i in range(len(points_flag)):
        p = points[points_flag[i]]
        if len(p) > 0:
            points_is_within_3d_box.append(p)
            box = bboxes[i]
            points_canonical, box_canonical = utils.points_to_canonical(p, box, args.oriangle)
            points_canonical, box_canonical = utils.lidar_to_shapenet(points_canonical, box_canonical)
            pts_name = '{}_{}_point_{}'.format(args.idx, args.category, i)
            box_name = '{}_{}_bbox_{}'.format(args.idx, args.category, i)
            utils.write_points(points_canonical, [output, pts_name])
            utils.write_bboxes(box_canonical, [output, box_name])

    points_is_within_3d_box = np.concatenate(points_is_within_3d_box, axis=0)
    points = points_is_within_3d_box

    utils.write_points(points, 'output/points')
    utils.write_bboxes(bboxes, 'output/bboxes')


if __name__ == "__main__":
    main()
