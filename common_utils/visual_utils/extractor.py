import os
import common_utils.cfgs as Config
import numpy as np
from visual_modul import open3d_vis_utils as V, io_utils as io
from visual_modul.calibration import Calibration


# labels extracted are processed and their location (to world coordinate), size (extended) and angle is different from original labels
def extract_tracking_scene(labelroot, calibroot, pointroot, outputroot, extend=1.3, begin=0, end=1, datatype=0,
                           threshold=0.85,
                           inference=True,
                           keep_world_coord=False):
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
    for index in range(begin, end):
        labeltxt = labeltxts[index]
        # used to transfer the coordinates to lidar's
        calibration = Calibration(os.path.join(calibroot, "{:04d}.txt".format(int(labeltxt.rstrip(".txt")))))

        # sceneId is int(labeltxt.rstrip(".txt"))
        sceneid = labeltxt.rstrip(".txt")
        cache_scenepoints_path = None
        cache_scenepoints = None
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
                    box = labellist[13:16] + [labellist[12], labellist[10], labellist[11]] + \
                          [labellist[16]]  # location,l, h, w,angle
                    extend_mtx = np.array([1, 1, 1, extend, extend, extend, 1]).reshape(1, -1)
                box = calibration.bbox_rect_to_lidar(np.array(box, dtype=np.float32).reshape(-1, len(box)) * extend_mtx) \
                    .reshape(len(box), )

                # get .bin (velodyne)
                pointspath = os.path.join(pointroot, sceneid, "{:06d}.bin".format(int(frameid)))
                if cache_scenepoints_path != pointspath:
                    cache_scenepoints_path = pointspath
                    try:
                        points = io.load_points(pointspath)
                        cache_scenepoints = points
                    except FileNotFoundError as fnf:
                        print("FileNotFoundError: no %s" % pointspath)
                        cache_scenepoints = None
                        continue
                else:
                    if cache_scenepoints is not None:
                        points = cache_scenepoints
                    else:
                        continue

                # extract and save points; save label in a txt for every TID
                extracted_points, _, _ = V.extract_object(points, box, keep_world_coord)
                if extracted_points.shape[0] == 0:
                    print("%s (id:%d) in scene %d has no points (no extraction)" %
                          (category, int(trajectoryid), int(sceneid)))
                    continue
                outputpath = os.path.join(outputroot, sceneid, "{}#{}".format(category, trajectoryid))
                name = "{:06d}".format(int(frameid))
                io.save_object_points(extracted_points, os.path.join(outputpath, "points"), name + ".npy")
                io.save_object_label(box, os.path.join(outputpath, "labels"),
                                     name + ".txt")  # save [location,l,h,w,angle], size is extended
                label = f.readline().rstrip("\n")

        print("Scene{} finished!".format(sceneid))


if __name__ == "__main__":
    cfg = Config.load_visual("extract_root")
    extract_tracking_scene(cfg["labelroot"], cfg["calibroot"], cfg["pointroot"],
                           cfg["outputroot"][bool(cfg["keep_world_coord"])],
                           begin=cfg["begin"], end=cfg["end"],
                           inference=bool(cfg["inference"]),
                           threshold=float(cfg["threshold"]),
                           extend=float(cfg["extend"]),
                           keep_world_coord=bool(cfg["keep_world_coord"]))
