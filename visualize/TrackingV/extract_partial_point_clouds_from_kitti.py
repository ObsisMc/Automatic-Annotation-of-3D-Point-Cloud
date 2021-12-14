import argparse
import numpy as np
import visualize.TrackingV.utils as utils
from visualize.TrackingV.calibration import Calibration

output = "output/tracking"


class pseduoargs():
    def __init__(self, idx, cate, oriangle):
        self.idx = idx
        self.category = cate
        self.oriangle = oriangle


def main(shrun=False, idx='0003', category='car', oriangle=False, calib="kitti/training/tracking/calib/",
         label="kitti/training/tracking/label_2/", velodyn="kitti/training/tracking/velodyne/"):
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

    label_path = label + '{}.txt'.format(args.idx)
    points_path = velodyn + "{}/".format(args.idx)
    bboxes, cates = utils.load_3d_boxes(label_path, args.category)  # with dontcare
    calib_path = calib + '{}.txt'.format(args.idx)
    calib = Calibration(calib_path)

    framen = utils.getFrameNumber(velodyn)
    framep = 0
    for i in range(1):
        points_path = points_path + '{:06}.bin'.format(i)
        points = utils.load_point_clouds(points_path)

        framebboxes = []
        framecates = []
        while True:
            if bboxes[framep][0] == i:
                if cates[framep] != "DontCare":
                    framebboxes.append(bboxes[framep])
                    framecates.append(cates[framep])
            else:
                break
            framep += 1
        if len(framebboxes) == 0:
            continue

        # load all object of frame i without dontcare
        framebboxes = np.array(framebboxes)
        framebboxestmp = calib.bbox_rect_to_lidar(framebboxes[:, 2:])

        corners3d = utils.boxes_to_corners_3d(framebboxestmp)
        points_flag = utils.is_within_3d_box(points, corners3d)

        points_is_within_3d_box = []
        for j in range(len(points_flag)):
            p = points[points_flag[j]]
            if len(p) > 0:
                points_is_within_3d_box.append(p)
                box = framebboxestmp[j]
                points_canonical, box_canonical = utils.points_to_canonical(p, box, args.oriangle)
                points_canonical, box_canonical = utils.lidar_to_shapenet(points_canonical, box_canonical)
                pts_name = '{}_{}_{}_{}point'.format(int(framebboxes[j][1]), framecates[j], i,
                                                     args.idx)  # tracking id, cate, frameid,sceneid
                box_name = '{}_{}_{}_{}bbox'.format(int(framebboxes[j][1]), framecates[j], i, args.idx)
                utils.write_points(points_canonical, [output, pts_name])
                utils.write_bboxes(box_canonical, [output, box_name])

        points_is_within_3d_box = np.concatenate(points_is_within_3d_box, axis=0)
        points = points_is_within_3d_box

        utils.write_points(points, 'output/points')
        utils.write_bboxes(framebboxes, 'output/bboxes')


if __name__ == "__main__":
    main()
