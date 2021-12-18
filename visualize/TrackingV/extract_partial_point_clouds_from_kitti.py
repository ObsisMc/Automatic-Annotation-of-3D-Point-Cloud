import argparse
import numpy as np
import visualize.TrackingV.utils as utils
from visualize.TrackingV.calibration import Calibration

prefix = "../Data/"
output = prefix + "Mydataset/"


class pseduoargs():
    def __init__(self, sceneid, oriangle, dilate, generror):
        self.sceneid = sceneid
        self.oriangle = oriangle
        self.dilate = dilate
        self.generror = generror


def main(shrun=False, sceneid='0003', oriangle=False, dilate=1, generror=False,
         calib=prefix + "kitti/training/tracking/calib/",
         label=prefix + "kitti/training/tracking/label_02_error/",
         velodyn=prefix + "kitti/training/tracking/velodyne/"):
    global output

    if not shrun:
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--sceneid', type=str, default='000936',
                            help='specify data index: {sceneid}.bin')
        parser.add_argument("--oriangle", action="store_true")
        args = parser.parse_args()
    else:
        args = pseduoargs(sceneid, oriangle, dilate, generror)

    label_path = label + '{}.txt'.format(args.sceneid)
    points_path = velodyn + "{}/".format(args.sceneid)
    bboxes, cates = utils.load_3d_boxes(label_path, dilate=args.dilate, error=generror)  # with dontcare
    calib_path = calib + '{}.txt'.format(args.sceneid)
    calib = Calibration(calib_path)

    framen = utils.getFrameNumber(points_path)
    framep = 0
    output = output + "{}/".format(args.sceneid)

    def extract(innerboxes, frameid, outpath, boxlabel=None):
        outpath = output + outpath
        framebboxestmp = calib.bbox_rect_to_lidar(innerboxes[:, 2:17])
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
                pidpath = "{}_{}/".format(framecates[j], int(innerboxes[j][1]))  # cate, pid
                pts_name = 'point{}'.format(frameid)  # tracking id, cate, frameid,sceneid
                box_name = 'bbox{}'.format(frameid)
                # generate label
                if boxlabel is not None:
                    labelpath = output + "label/"
                    labelname = "{}_{}.txt".format(framecates[j], int(innerboxes[j][1]))
                    utils.write_labels(errorlabel[j], frameid, [labelpath, labelname])

                utils.write_points(points_canonical, [outpath + pidpath, pts_name])
                utils.write_bboxes(box_canonical, [outpath + pidpath, box_name])

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
        # framebboxes‘ len is 9 or 13
        framebboxes = np.array(framebboxes)
        extract(framebboxes[:, :9], i, "groundtruth/")

        # # 如果要生成偏移数据
        if generror:
            errorlabel = np.c_[
                (framebboxes[:, 9:12] - framebboxes[:, 2:5]), (framebboxes[:, 12] - framebboxes[:, 8])]
            framebboxes[:, 2:5] = framebboxes[:, 9:12]
            framebboxes[:, 8] = framebboxes[:, 12]
            extract(framebboxes[:, :9], i, "error/", errorlabel)


if __name__ == "__main__":
    main()
