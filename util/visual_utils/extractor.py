import os

import numpy as np
from visual_modul import open3d_vis_utils as V, io_utils as io
from visual_modul.calibration import Calibration


def extract_tracking_scene(labelroot, calibroot, pointroot, outputroot, maxn=1, datatype=0, threshold=0.7,
                           inference=True):
    """
    All directory structure is the same as kitti-tracking data.
    it contains frameId and trajectoryId which must return.
    datatype: 0 for validation (with all labels), 1 for test

    ouotput structure:
    root
      |-> 0000 // scene
            |-> Car#1 // type and trajectoryId
                  |-> points
                    |-> 000000.npy // points cloud in frameId
                  |-> labels
                    |-> 000000.txt

    """

    # load data
    labeltxts = sorted(os.listdir(labelroot), key=lambda x: int(x.rstrip(".txt")))
    n = 0
    for labeltxt in labeltxts:
        if n == maxn:
            break
        # used to transfer the coordinates to lidar's
        calibration = Calibration(os.path.join(calibroot, "{:04d}.txt".format(int(labeltxt.rstrip(".txt")))))

        # sceneId is int(labeltxt.rstrip(".txt"))
        sceneid = labeltxt.rstrip(".txt")
        with open(os.path.join(labelroot, labeltxt), "r") as f:
            label = f.readline().rstrip("\n")
            while label and label != "":
                labellist = label.split(" ")
                label = f.readline().rstrip("\n")

                if inference:
                    confidence = labellist[17]
                    if float(confidence) < threshold:
                        continue

                # get necessary info
                if datatype == 0:
                    frameid, trajectoryid, category = labellist[0], labellist[1], labellist[2]
                    if category == "DontCare":
                        continue
                    box = labellist[13:16] + [labellist[12], labellist[10], labellist[11]] + [labellist[16]]
                box = calibration.bbox_rect_to_lidar(np.array(box, dtype=np.float32).reshape(-1, len(box))) \
                    .reshape(len(box), )

                # get .bin (velodyne)
                pointspath = os.path.join(pointroot, sceneid, "{:06d}.bin".format(int(frameid)))
                points = io.load_points(pointspath)

                # extract and save points; save label in a txt for every TID
                extracted_points, _ = V.extract_object(points, box)
                outputpath = os.path.join(outputroot, sceneid, "{}#{}".format(category, trajectoryid))
                name = "{:06d}".format(int(frameid))
                io.save_object_points(extracted_points, os.path.join(outputpath, "points"), name + ".npy")
                io.save_object_label(box[[0, 1, 2, 6]], os.path.join(outputpath, "labels"),
                                     name + ".txt")  # save location and angle
                label = f.readline().rstrip("\n")

        print("Scene{} finished!".format(sceneid))
        n += 1


if __name__ == "__main__":
    labelroot = "/home/zrh/Data/kitti/data_tracking_label_2/training/label_02/"
    calibroot = "/home/zrh/Data/kitti/data_tracking_calib/training/calib/"
    pointroot = "/home/zrh/Data/kitti/data_tracking_velodyne/"
    outputroot = "/home/zrh/Data/kitti/tracking/extracted_points"
    extract_tracking_scene(labelroot, calibroot, pointroot, outputroot, inference=False)