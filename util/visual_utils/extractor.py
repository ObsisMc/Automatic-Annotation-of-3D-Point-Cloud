import os

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

                # get necessary info
                if datatype == 0:
                    frameid, trajectoryid, category, confidence = labellist[0], labellist[1], labellist[2], label[17]
                    box = labellist[13:16] + [labellist[12], labellist[10], labellist[11]] + [labellist[16]]
                if int(confidence) < threshold:
                    continue
                box = calibration.bbox_rect_to_lidar(box)

                # get .bin (velodyne)
                pointspath = os.path.join(pointroot, sceneid, "{:06d}.bin".format(int(frameid)))
                points = io.load_points(pointspath)

                # extract and save points
                extracted_points, _ = V.extract_object(points, box)
                outputpath = os.path.join(outputroot, sceneid, "{}#{}".format(category, trajectoryid))
                npyname = "{:06d}.npy".format(int(frameid))
                io.save_object(extracted_points, outputpath, npyname)


if __name__ == "__main__":
    extract_tracking_scene()
