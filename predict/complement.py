import os
import predict
import numpy as np
import cv2

data_dir = "/home2/lie/InnovativePractice2/data/kitti/tracking/extracted_08_1"


def get_trajectory_path(data_dir):
    """
    :param data_dir: the directory of the extracted data
    :return: the path of all the trajectory files. ex: .../Car#1
    """
    scene_list = os.listdir(data_dir)
    trajectory_path = []
    for scene in scene_list:
        scene_path = os.path.join(data_dir, scene)
        trajectory_list = os.listdir(scene_path)
        for trajectory in trajectory_list:
            trajectory_path.append(os.path.join(scene_path, trajectory))
    return trajectory_path


def get_scene_list(data_dir):
    """
    :param data_dir: the directory of the extracted data
    :return: the list of all the scene names. ex: .../0000
    """
    scene_names = os.listdir(data_dir)
    scene_names.sort()
    scene_list = []
    for scene in scene_names:
        scene_list.append(os.path.join(data_dir, scene))
    return scene_list


def find_longest_continuous(trajectory_path):
    """
    :param trajectory_path: the path of the trajectory files
    :return: the beginning and the end of the longest continuous trajectory
    """
    label_names = os.listdir(trajectory_path + "labels")
    label_names.sort()
    begin_frame = int(label_names[0].split(".")[0])
    end_frame = int(label_names[0].split(".")[0])
    result_begin = 0
    result_end = 0
    for name in label_names[1:]:
        frame = int(name.split(".")[0])
        if frame - end_frame == 1:
            end_frame = frame
        else:
            if end_frame - begin_frame > result_end - result_begin:
                result_begin = begin_frame
                result_end = end_frame
            begin_frame = frame
            end_frame = frame
    return result_begin, result_end


def get_all_other_labels(scene, trajectory):
    """
    :param scene: the path of the scene
    :param trajectory: the path of the trajectory
    :return: the list of all other label paths of the same kind of thing within the scene.
    """
    category = os.path.basename(trajectory).split('#')[0]
    labels = []
    trajectory_names = os.listdir(scene)
    trajectory_paths = [os.path.join(scene, name) for name in trajectory_names if name.startswith(category)]
    for path in trajectory_paths:
        if path != trajectory:
            label_names = os.listdir(path + "labels")
            for label_name in label_names:
                labels.append(os.path.join(path, "labels", label_name))
    return labels


if __name__ == "__main__":
    trajectory_paths = get_trajectory_path(data_dir)
    # Complement within the trajectory first
    # Process each trajectory
    for trajectory in trajectory_paths:
        # Initialize l, h, w of the trajectory
        l, h, w = 0, 0, 0
        label_names = os.listdir(trajectory + "labels")
        label_names.sort()
        # Get the frame range of the trajectory
        first_frame = int(label_names[0].split(".")[0])
        last_frame = int(label_names[-1].split(".")[0])
        # Complement the trajectory until it is continuous
        while True:
            begin, end = find_longest_continuous(trajectory)
            if begin == first_frame and end == last_frame:
                break
            # If there is no continuous trajectory, find the first two frames
            if begin == end:
                label_names = os.listdir(trajectory + "labels")
                label_names.sort()
                first_frame = int(label_names[0].split(".")[0])
                second_frame = int(label_names[1].split(".")[0])
                # Get x, y, z, l, h, w, theta of the two frames
                label_name = str(first_frame).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "labels", label_name)
                with open(label_path, "r") as f:
                    line = f.readline()
                    data = line.split(" ")
                    x1, y1, z1, theta1 = float(data[0]), float(data[1]), float(data[2]), float(data[6])
                    l, h, w = float(data[3]), float(data[4]), float(data[5])
                label_name = str(second_frame).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "labels", label_name)
                with open(label_path, "r") as f:
                    line = f.readline()
                    data = line.split(" ")
                    x2, y2, z2, theta2 = float(data[0]), float(data[1]), float(data[2]), float(data[6])
                # Predict the position of the frame after the first frame
                x = x1 + (x2 - x1) / (last_frame - first_frame)
                y = y1 + (y2 - y1) / (last_frame - first_frame)
                z = z1 + (z2 - z1) / (last_frame - first_frame)
                theta = theta1 + (theta2 - theta1) / (last_frame - first_frame)
                # TODO: Extract the point cloud and invoke the network, remember add the point cloud to points
                # Write the predicted position back to the label file
                label_name = str(first_frame + 1).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "labels", label_name)
                with open(label_path, "w") as f:
                    f.write(
                        str(x) + " " + str(y) + " " + str(z) + " " + str(l) + " " + str(h) + " " + str(w) + " " + str(
                            theta))
                continue
            position_list = []
            # Get x, y, z, theta of the longest continuous trajectory
            for i in range(begin, end + 1):
                label_name = str(i).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "labels", label_name)
                with open(label_path, "r") as f:
                    line = f.readline()
                    data = line.split(" ")
                    position_list.append([float(data[0]), float(data[1]), float(data[2]), float(data[6])])
                    l, h, w = float(data[3]), float(data[4]), float(data[5])
            # Predict the trajectory backward first
            if begin != first_frame:
                pred_x = predict.time_predict(position_list[:][0].reverse())
                pred_y = predict.time_predict(position_list[:][1].reverse())
                pred_z = predict.time_predict(position_list[:][2].reverse())
                pred_theta = predict.time_predict(position_list[:][3].reverse())
                # TODO: Extract the point cloud and invoke the network, remember add the point cloud to points
                # Write the predicted position back to the label file
                label_name = str(begin - 1).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "labels", label_name)
                with open(label_path, "w") as f:
                    f.write(
                        str(pred_x) + " " + str(pred_y) + " " + str(pred_z) + " " + str(l) + " " + str(h) + " " + str(
                            w) + " " + str(pred_theta))
            # Predict the trajectory forward
            if end != last_frame:
                pred_x = predict.time_predict(position_list[:][0])
                pred_y = predict.time_predict(position_list[:][1])
                pred_z = predict.time_predict(position_list[:][2])
                pred_theta = predict.time_predict(position_list[:][3])
                # TODO: Extract the point cloud and invoke the network, remember add the point cloud to points
                # Write the predicted position back to the label file
                label_name = str(end + 1).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "labels", label_name)
                with open(label_path, "w") as f:
                    f.write(
                        str(pred_x) + " " + str(pred_y) + " " + str(pred_z) + " " + str(l) + " " + str(h) + " " + str(
                            w) + " " + str(pred_theta))
    # Predict both ends of the trajectory and merge overlapping trajectories.
    scene_paths = get_scene_list(data_dir)
    for scene in scene_paths:
        trajectory_names = os.listdir(scene)
        trajectory_names.sort()
        trajectory_paths = [os.path.join(scene, trajectory_name) for trajectory_name in trajectory_names]
        for trajectory in trajectory_paths:
            # Initialize l, h, w of the trajectory
            l, h, w = 0, 0, 0
            label_names = os.listdir(trajectory + "labels")
            label_names.sort()
            # Get the frame range of the trajectory
            first_frame = int(label_names[0].split(".")[0])
            last_frame = int(label_names[-1].split(".")[0])
            # If the trajectory is too short, skip it
            if last_frame - first_frame < 3:
                continue
            position_list = []
            # Get x, y, z, theta of the trajectory
            for i in range(first_frame, last_frame + 1):
                label_name = str(i).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "labels", label_name)
                with open(label_path, "r") as f:
                    line = f.readline()
                    data = line.split(" ")
                    position_list.append([float(data[0]), float(data[1]), float(data[2]), float(data[6])])
                    l, h, w = float(data[3]), float(data[4]), float(data[5])
            # Predict the trajectory backward first
            if first_frame != 0:
                pred_x = predict.time_predict(position_list[:][0].reverse())
                pred_y = predict.time_predict(position_list[:][1].reverse())
                pred_z = predict.time_predict(position_list[:][2].reverse())
                pred_theta = predict.time_predict(position_list[:][3].reverse())
                # TODO: Extract the point cloud and invoke the network, remember add the point cloud to points
                add = False
                if add:
                    # Check if the predicted trajectory is overlapping with the previous trajectory
                    label_paths = get_all_other_labels(scene, trajectory)
                    for label_path in label_paths:
                        with open(label_path, "r") as f:
                            line = f.readline()
                            data = line.split(" ")
                            # If the distance between the predicted box and the previous box is less than 0.1m and theta is less than 0.1rad, merge the two trajectories
                            if (pred_x - float(data[0])) ** 2 + (pred_y - float(data[1])) ** 2 + (pred_z - float(data[2])) ** 2 < 0.01 and abs(
                                    pred_theta - float(data[6])) < 0.1:
                                # TODO: Merge the two trajectories
                    # Write the predicted position back to the label file
                    label_name = str(first_frame - 1).zfill(6) + ".txt"
                    label_path = os.path.join(trajectory + "labels", label_name)
                    with open(label_path, "w") as f:
                        f.write(
                            str(pred_x) + " " + str(pred_y) + " " + str(pred_z) + " " + str(l) + " " + str(
                                h) + " " + str(
                                w) + " " + str(pred_theta))

