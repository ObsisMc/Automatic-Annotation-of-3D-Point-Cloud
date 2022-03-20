import os

import numpy as np
from visual_modul import open3d_vis_utils as V, io_utils as io
from visual_modul.calibration import Calibration


def extract_tracking_scene(labelroot, calibroot, pointroot, outputroot, datatype=0, threshold=0.7):
    """
    All directory structure is the same as kitti-tracking data.
    it contains frameId and trajectoryId which must return.
    datatype: 0 for validation (with all labels), 1 for test

    ouotput structure:
    root
      |-> 0000 // scene
            |-> Car#1 // type and trajectoryId
                  |-> 000000.npy // points cloud in frameId
    """
    # used to transfer the coordinates to lidar's
    calibration = Calibration(calibroot)

    # load data
    labeltxts = sorted(os.listdir(labelroot), key=lambda x: int(x.rstrip(".txt")))
    for labeltxt in labeltxts:
        # sceneId is int(labeltxt.rstrip(".txt"))
        sceneid = labeltxt.rstrip(".txt")
        with open(os.path.join(labelroot, labeltxt), "r") as f:
            label = f.readline().rstrip("\n")
            while label and label != "":
                labellist = label.split(" ")
                label = f.readline().rstrip("\n")

                confidence = labellist[17]
                if float(confidence) < threshold:
                    continue

                # get necessary info
                if datatype == 0:
                    frameid, trajectoryid, category = labellist[0], labellist[1], labellist[2]
                    box = labellist[13:16] + [labellist[12], labellist[10], labellist[11]] + [labellist[16]]
                box = calibration.bbox_rect_to_lidar(np.array(box, dtype=np.float32).reshape(-1, len(box))) \
                    .reshape(len(box), )

                # get .bin (velodyne)
                pointspath = os.path.join(pointroot, sceneid, "{:06d}.bin".format(int(frameid)))
                points = io.load_points(pointspath)

                # extract and save points
                extracted_points, _ = V.extract_object(points, box)
                outputpath = os.path.join(outputroot, sceneid, "{}#{}".format(category, trajectoryid))
                npyname = "{:06d}.npy".format(int(frameid))
                io.save_object(extracted_points, outputpath, npyname)
                label = f.readline().rstrip("\n")

        print("Scene{} finished!".format(sceneid))


if __name__ == "__main__":
    labelroot = "/home2/lie/InnovativePractice2/AB3DMOT/results/pvrcnn_0_9_test/data"
    calibroot = "/public_dataset/kitti/tracking/data_tracking_calib/training/calib/0000.txt"
    pointroot = "/public_dataset/kitti/tracking/data_tracking_velodyne/training/velodyne"
    outputroot = "/home2/lie/InnovativePractice2/data/kitti/tracking/extracted_points"
    extract_tracking_scene(labelroot, calibroot, pointroot, outputroot)
