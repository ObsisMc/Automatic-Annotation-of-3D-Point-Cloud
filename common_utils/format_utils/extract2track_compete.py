import os
import numpy as np
from common_utils.visual_utils.visual_modul.calibration import Calibration

label_dir = "labels_test"


def preProcess(extract_root, scene: str, type):
    path = os.path.join(extract_root, scene)
    trajs_list = os.listdir(path)
    framen = 0
    for i in range(len(trajs_list)):
        if trajs_list[i].split("#")[0] not in type:
            continue
        traj_label_path = os.path.join(path, trajs_list[i], label_dir)
        labels_list = os.listdir(traj_label_path)
        for label_txt in labels_list:
            frame = int(label_txt.rstrip(".txt"))
            framen = frame if frame > framen else framen
    return framen


def extractToTracking(extract_root, output_root, calib_path, extend=1.3, type=("Car")):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    scene_list = sorted(os.listdir(extract_root), key=lambda x: int(x))
    for i in range(len(scene_list)):
        scene = scene_list[i]
        # load calib
        calibration = Calibration(os.path.join(calib_path, scene + ".txt"))

        framen = preProcess(extract_root, scene, type)
        slots = [[] for _ in range(framen + 1)]

        path = os.path.join(extract_root, scene)
        trajs_list = os.listdir(path)
        for j in range(len(trajs_list)):
            category = trajs_list[j].split("#")[0]
            tid = trajs_list[j].split("#")[1]

            if category not in type:
                continue
            traj = trajs_list[j]
            label_path = os.path.join(path, traj, label_dir)
            label_list = os.listdir(label_path)
            for label_txt in label_list:
                with open(os.path.join(label_path, label_txt), "r") as f:
                    label = f.readline().split(" ")

                    frame = [str(int(label_txt.rstrip(".txt")))]
                    kitti_size = [str(float(label[4]) / extend), str(float(label[5]) / extend),
                                  str(float(label[3]) / extend)]  # size from extend*[l, h, w] to [h, w, l]
                    kitti_loc = calibration.lidar_to_rect(
                        np.array(label[:3], dtype=np.float32).reshape(-1, 3)).reshape(
                        -1, ).tolist()  # to camera coordinate
                    kitti_loc_str = [str(i) for i in kitti_loc]
                    kitti_angle = [str(-float(label[6]) - np.pi / 2)]
                    bbox2d = ["0.", "0.", "0.", "0."]
                    alpha = ["0."]
                    truncated, occluded = ["0."], ["0."]

                    label_record = frame + [tid] + [category] + truncated + occluded + alpha + bbox2d + \
                                   kitti_size + kitti_loc_str + kitti_angle
                    slots[int(frame[0])].append(" ".join(label_record))

        with open(os.path.join(output_root, scene + ".txt"), 'w') as f:
            for i in range(len(slots)):
                for j in range(len(slots[i])):
                    f.write(slots[i][j])
                    f.write("\n")


if __name__ == "__main__":
    # TODO: only retains Car is enough?
    extract_root = "/home/zrh/Data/kitti/tracking/extracted_points"
    output_root = os.path.join("/home/zrh/Data/kitti/tracking", "extract2track", "label_02")
    calib_path = "/home/zrh/Data/kitti/data_tracking_calib/training/calib"
    extractToTracking(extract_root, output_root, calib_path, extend=1.3)
