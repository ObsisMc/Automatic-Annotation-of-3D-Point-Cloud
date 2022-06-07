import os
import sys

import numpy as np

sys.path.append('..')
from visual_utils.visual_modul.calibration import Calibration

extract_dir = '/home2/lie/InnovativePractice2/data/kitti/tracking/extracted_08_2_test'
output_dir = '/home2/lie/InnovativePractice2/AB3DMOT/results/pv_rcnn_epoch_8369_08_test/object_format_final'
calib_dir = '/public_dataset/kitti/tracking/data_tracking_calib/training/calib'
scene_split = [0, 154, 601, 834, 978, 1292, 1589, 1859, 2659, 3049, 3852, 4146, 4519, 4597, 4937, 5043, 5419, 5628,
               5773, 6112, 7171, 8008]


def extract_to_track(extract_dir, output_dir, scene_split):
    scene_list = os.listdir(extract_dir)
    for scene_id in scene_list:
        print("Processing scene {}".format(scene_id))
        calibration = Calibration(os.path.join(calib_dir, "{:04d}.txt".format(int(scene_id))))
        traj_list = os.listdir(os.path.join(extract_dir, scene_id))
        for traj_id in traj_list:
            label_list = os.listdir(os.path.join(extract_dir, scene_id, traj_id, 'labels'))
            for label_name in label_list:
                with open(os.path.join(extract_dir, scene_id, traj_id, 'labels', label_name), 'r') as f:
                    data = f.readline().split(' ')
                frame_num = int(label_name.rstrip('.txt')) + scene_split[int(scene_id)]
                with open(os.path.join(output_dir, '{:06d}.txt'.format(frame_num)), 'a') as f:
                    type = traj_id.split('#')[0]
                    dimensions = data[5] + ' ' + data[4] + ' ' + data[3]
                    lidar_box = np.array([[float(data[0]), float(data[1]), float(data[2]), float(data[3]) * 1.3,
                                           float(data[4]) * 1.3, float(data[5]) * 1.3, float(data[6])]])
                    rect_box = calibration.lidar_to_bbox_rect(lidar_box, 1.3)
                    location = str(rect_box[0][0]) + ' ' + str(rect_box[0][1]) + ' ' + str(rect_box[0][2])
                    angle = str(rect_box[0][6])
                    f.write(type + ' ' + '0 0 0 0 0 0 0 ' + dimensions + ' ' + location + ' ' + angle + ' 1\n')
    for i in range(scene_split[-1]):
        # If {.06d}.txt does not exist, create it
        if not os.path.exists(os.path.join(output_dir, '{:06d}.txt'.format(i))):
            open(os.path.join(output_dir, '{:06d}.txt'.format(i)), 'a').close()


if __name__ == '__main__':
    extract_to_track(extract_dir, output_dir, scene_split)
