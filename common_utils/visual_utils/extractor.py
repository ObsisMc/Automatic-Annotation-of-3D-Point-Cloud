import os
import common_utils.cfgs as Config
import numpy as np
from visual_modul import open3d_vis_utils as V, io_utils as io
from visual_modul.calibration import Calibration
from visual_modul.oxst_projector import OxstProjector


# labels extracted are processed and their location (to world coordinate), size (extended) and angle is different from original labels
def extract_tracking_scene(labelroot, calibroot, pointroot, outputroot, extend=1.3, begin=0, end=1, datatype=0,
                           threshold=0.85,
                           inference=True,
                           point_transform="default",
                           oxstroot=None):
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
    # init
    point_transform = point_transform.lower()

    # load data
    labeltxts = sorted(os.listdir(labelroot), key=lambda x: int(x.rstrip(".txt")))
    for index in range(begin, end):
        labeltxt = labeltxts[index]
        # sceneId is int(labeltxt.rstrip(".txt"))
        sceneid = labeltxt.rstrip(".txt")

        # used to transfer the coordinates to lidar's
        calibration = Calibration(os.path.join(calibroot, "{:04d}.txt".format(int(sceneid))))

        # oxst
        oxst_projector = None
        oxsts = None
        if point_transform == "earth":
            oxst_projector = OxstProjector()
            oxsts = io.load_oxst_tracking_scene(os.path.join(oxstroot, "{:04d}.txt".format(int(sceneid))))

        # PointTransformer
        point_transformer = PointTransformer(point_transform, oxst_projector)

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
                extracted_points, _, _ = V.extract_object(points, box)

                # change points' coordinates
                if point_transform == "default":
                    extracted_points, new_box = point_transformer.transform(extracted_points, box)
                elif point_transform == "canonical":
                    extracted_points, new_box = point_transformer.transform(extracted_points, box)
                elif point_transform == "earth":
                    extracted_points, new_box = point_transformer.transform(extracted_points, oxsts[int(frameid)], box)

                if extracted_points.shape[0] == 0:
                    print("%s (id:%d) in scene %d has no points (no extraction)" %
                          (category, int(trajectoryid), int(sceneid)))
                    continue
                outputpath = os.path.join(outputroot, sceneid, "{}#{}".format(category, trajectoryid))
                name = "{:06d}".format(int(frameid))
                io.save_object_points(extracted_points, os.path.join(outputpath, "points"), name + ".npy")
                io.save_object_label(new_box, os.path.join(outputpath, "labels"),
                                     name + ".txt")  # save [location,l,h,w,angle], size is extended
                label = f.readline().rstrip("\n")

        print("Scene{} finished!".format(sceneid))


def extract_single_object(scene_points: np.ndarray, calib: Calibration, box: str, extend=1,
                          point_transform="canonical"):
    """
    @params box, a string about a single tracking label
    """
    calib_box = calib.bbox_rect_to_lidar(io.trans_track_txt_to_calib_format(box, extend)).reshape(-1, )
    extracted_points, _, _ = V.extract_object(scene_points, calib_box)

    point_transformer = PointTransformer(point_transform)
    extracted_points, new_box = point_transformer.transform(extracted_points, calib_box)
    return extracted_points


class PointTransformer:
    __DEFAULT = "default"
    __CANONICAL = "canonical"
    __EARTH = "earth"

    def __init__(self, handle_type: str = 'default', oxst_projector: OxstProjector = None):
        self.handle_type = handle_type.lower()
        self.transform = self.transform_handler()

        self.oxst_projector = oxst_projector

    def transform_handler(self):
        def handler(*arg):
            if self.handle_type == self.__DEFAULT:
                return self.default_handle(*arg)
            elif self.handle_type == self.__CANONICAL:
                return self.canonicalize(*arg)
            elif self.handle_type == self.__EARTH:
                return self.to_earth_pose(*arg)

            raise "Unknown handle type"

        return handler

    @staticmethod
    def canonicalize(points: np.ndarray, box: np.ndarray):
        # translation
        center = box[0:3].reshape(1, -1)
        points_translated = (points - center).reshape(1, -1)

        # rotation
        points_canonical = PointTransformer.rotate_points_along_z(points_translated.reshape(1, -1, 3), -box[-1])
        return points_canonical, box

    def to_earth_pose(self, points: np.ndarray, oxst_config, box: np.ndarray):
        assert self.oxst_projector is not None
        self.oxst_projector.init_oxst(oxst_config)
        points = self.oxst_projector.lidar_to_pose(points)
        new_box = np.copy(box)
        new_box[0], new_box[1], new_box[2], yaw = self.oxst_projector.get_offset()
        new_box[5] += np.pi / 2 - new_box[5] + yaw
        return points, new_box

    @staticmethod
    def default_handle(points: np.ndarray, box: np.ndarray):
        return points, box

    @staticmethod
    def rotate_points_along_z(points, angle):
        """
        Args:
            points: (B, N, 3)
            angle: (B), angle along z-axis, angle increases x ==> y
        Returns:
        """
        cosa = np.cos(angle)
        sina = np.sin(angle)
        ones = np.ones_like(angle, dtype=np.float32)
        zeros = np.zeros_like(angle, dtype=np.float32)
        rot_matrix = np.array(
            [[cosa, sina, zeros],
             [-sina, cosa, zeros],
             [zeros, zeros, ones]]
        )
        points_rot = points @ rot_matrix
        return points_rot


if __name__ == "__main__":
    cfg = Config.load_visual("extract_root")
    extract_tracking_scene(cfg["labelroot"], cfg["calibroot"], cfg["pointroot"],
                           cfg["outputroot"][cfg["point_transform"]],
                           begin=cfg["begin"], end=cfg["end"],
                           inference=bool(cfg["inference"]),
                           threshold=float(cfg["threshold"]),
                           extend=float(cfg["extend"]),
                           point_transform=cfg["point_transform"],
                           oxstroot=cfg["oxstroot"]
                           )
