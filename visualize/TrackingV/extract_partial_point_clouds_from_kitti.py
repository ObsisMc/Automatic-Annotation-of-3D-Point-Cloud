import argparse
import numpy as np
import visualize.TrackingV.utils as utils
from visualize.TrackingV.calibration import Calibration

output = "output/tracking/"


class pseduoargs():
    def __init__(self, sceneid, oriangle):
        self.sceneid = sceneid
        self.oriangle = oriangle


def main(shrun=False, sceneid='0003', oriangle=False, calib="kitti/training/tracking/calib/",
         label="kitti/training/tracking/label_2/", velodyn="kitti/training/tracking/velodyne/"):
    global output
    if not shrun:
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--sceneid', type=str, default='000936',
                            help='specify data index: {sceneid}.bin')
        parser.add_argument("--oriangle", action="store_true")
        args = parser.parse_args()
    else:
        args = pseduoargs(sceneid, oriangle)

    label_path = label + '{}.txt'.format(args.sceneid)
    points_path = velodyn + "{}/".format(args.sceneid)
    bboxes, cates = utils.load_3d_boxes(label_path)  # with dontcare
    calib_path = calib + '{}.txt'.format(args.sceneid)
    calib = Calibration(calib_path)

    framen = utils.getFrameNumber(points_path)
    framep = 0
    output = output + "{}/".format(args.sceneid)
    for i in range(framen):
        framepoints_path = points_path + '{:06}.bin'.format(i)
        points = utils.load_point_clouds(framepoints_path)

        framebboxes = []
        framecates = []
        while framep < len(bboxes):
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
                pidpath = "{}_{}/".format(framecates[j], int(framebboxes[j][1]))  # cate, pid
                pts_name = 'point{}'.format(i)  # tracking id, cate, frameid,sceneid
                box_name = 'bbox{}'.format(i)
                utils.write_points(points_canonical, [output + pidpath, pts_name])
                utils.write_bboxes(box_canonical, [output + pidpath, box_name])

        points_is_within_3d_box = np.concatenate(points_is_within_3d_box, axis=0)
        points = points_is_within_3d_box


if __name__ == "__main__":
    main()
