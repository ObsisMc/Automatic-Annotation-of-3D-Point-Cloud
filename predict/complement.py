import os
import predict
import numpy as np
import cv2
import shutil
import sys
sys.path.append("..")
from common_utils.visual_utils.extractor import extract_single_object
from common_utils.visual_utils.visual_modul.io_utils import load_points
from common_utils.visual_utils.visual_modul.calibration import Calibration
from OurNet.NewTest import test as net_predict

data_dir = "/home2/lie/InnovativePractice2/data/kitti/tracking/extracted_08_2_test"
velodyne_dir = "/public_dataset/kitti/tracking/data_tracking_velodyne/training/velodyne"
calib_dir = "/public_dataset/kitti/tracking/data_tracking_calib/training/calib"


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
    label_names = os.listdir(trajectory_path + "/labels")
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
            begin_frame = frame
            end_frame = frame
        if end_frame - begin_frame > result_end - result_begin:
            result_begin = begin_frame
            result_end = end_frame
    return result_begin, result_end


def get_all_other_labels(scene, trajectory, frame):
    """
    :param scene: the path of the scene
    :param trajectory: the path of the trajectory
    :param frame: the frame number
    :return: the list of all other label paths of the same kind of thing and same frame number within the scene.
    """
    category = os.path.basename(trajectory).split('#')[0]
    labels = []
    trajectory_names = os.listdir(scene)
    trajectory_paths = [os.path.join(scene, name) for name in trajectory_names if name.startswith(category)]
    for path in trajectory_paths:
        if path != trajectory:
            label_names = os.listdir(path + "/labels")
            for label_name in label_names:
                if label_name.split(".")[0] == str(frame).zfill(6):
                    labels.append(os.path.join(path, "labels", label_name))
    return labels


def move_files(trajectory1, trajectory2):
    """
    Move all labels and points from trajectory1 to trajectory2.
    """
    for label in os.listdir(trajectory1 + "/labels"):
        shutil.move(os.path.join(trajectory1 + "/labels", label), os.path.join(trajectory2 + "/labels", label))
    for point in os.listdir(trajectory1 + "/points"):
        shutil.move(os.path.join(trajectory1 + "/points", point), os.path.join(trajectory2 + "/points", point))


def extract_points(scene, frame_num, box):
    """
    :param scene: the path of the scene
    :param frame_num: the frame number
    :param box: the information of the bbox, tracking format, a string
    :return: the point cloud
    """
    scene_num = os.path.basename(scene)
    velodyne_path = os.path.join(velodyne_dir, scene_num, str(frame_num).zfill(6) + ".bin")
    calib_path = os.path.join(calib_dir, scene_num + ".txt")
    scene_points = load_points(velodyne_path)[:, :3]
    calib = Calibration(calib_path)
    extracted_points = extract_single_object(scene_points, calib, box, 1.3)
    return extracted_points


if __name__ == "__main__":
    trajectory_paths = get_trajectory_path(data_dir)
    # Complement within the trajectory first
    # Process each trajectory
    for trajectory in trajectory_paths:
        print("Processing trajectory: " + trajectory)
        # Initialize l, h, w of the trajectory
        l, h, w = 0, 0, 0
        label_names = os.listdir(trajectory + "/labels")
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
                label_names = os.listdir(trajectory + "/labels")
                if len(label_names) == 1:
                    break
                label_names.sort()
                first_frame = int(label_names[0].split(".")[0])
                second_frame = int(label_names[1].split(".")[0])
                # Get x, y, z, l, h, w, theta of the two frames
                label_name = str(first_frame).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "/labels", label_name)
                with open(label_path, "r") as f:
                    line = f.readline()
                    data = line.split(" ")
                    x1, y1, z1, theta1 = float(data[0]), float(data[1]), float(data[2]), float(data[6])
                    l, h, w = float(data[3]), float(data[4]), float(data[5])
                label_name = str(second_frame).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "/labels", label_name)
                with open(label_path, "r") as f:
                    line = f.readline()
                    data = line.split(" ")
                    x2, y2, z2, theta2 = float(data[0]), float(data[1]), float(data[2]), float(data[6])
                # Predict the position of the frame after the first frame
                x = x1 + (x2 - x1) / (last_frame - first_frame)
                y = y1 + (y2 - y1) / (last_frame - first_frame)
                z = z1 + (z2 - z1) / (last_frame - first_frame)
                theta = theta1 + (theta2 - theta1) / (last_frame - first_frame)
                # Extract the point cloud and invoke the network, remember add the point cloud to points
                scene = os.path.dirname(trajectory)
                box = "0 0 0 0 0 0 0 0 0 0 " + str(h) + " " + str(w) + " " + str(l) + " " + str(x) + " " + str(
                    y) + " " + str(z) + " " + str(theta)
                points = extract_points(scene, first_frame + 1, box).reshape(-1, 3)
                points_name = str(first_frame).zfill(6) + ".npy"
                points_path = os.path.join(trajectory + "/points", points_name)
                output = net_predict(np.load(points_path).reshape(-1, 3), points)
                output = output.view(-1).cpu().detach().numpy()
                if output[0] > 0.5:
                    x += output[1]
                    y += output[2]
                    z += output[3]
                    theta += output[4]
                # Save the points
                box = "0 0 0 0 0 0 0 0 0 0 " + str(h) + " " + str(w) + " " + str(l) + " " + str(x) + " " + str(
                y) + " " + str(z) + " " + str(theta)
                points = extract_points(scene, first_frame + 1, box)
                points_name = str(first_frame + 1).zfill(6) + ".npy"
                points_path = os.path.join(trajectory + "/points", points_name)
                np.save(points_path, points)
                # Write the predicted position back to the label file
                label_name = str(first_frame + 1).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "/labels", label_name)
                with open(label_path, "w") as f:
                    f.write(
                        str(x) + " " + str(y) + " " + str(z) + " " + str(l) + " " + str(h) + " " + str(w) + " " + str(
                            theta))
                print("Add frame " + str(first_frame + 1) + " to trajectory " + trajectory)
                continue
            position_list = []
            # Get x, y, z, theta of the longest continuous trajectory
            for i in range(begin, end + 1):
                label_name = str(i).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "/labels", label_name)
                with open(label_path, "r") as f:
                    line = f.readline()
                    data = line.split(" ")
                    position_list.append([float(data[0]), float(data[1]), float(data[2]), float(data[6])])
                    l, h, w = float(data[3]), float(data[4]), float(data[5])
            position_list = np.array(position_list)
            # Predict the trajectory backward first
            if begin != first_frame:
                pred_x = predict.time_predict(np.flipud(position_list[:, 0]))
                pred_y = predict.time_predict(np.flipud(position_list[:, 1]))
                pred_z = predict.time_predict(np.flipud(position_list[:, 2]))
                pred_theta = predict.time_predict(np.flipud(position_list[:, 3]))
                # Extract the point cloud and invoke the network, remember add the point cloud to points
                scene = os.path.dirname(trajectory)
                box = "0 0 0 0 0 0 0 0 0 0 " + str(h) + " " + str(w) + " " + str(l) + " " + str(pred_x) + " " + str(
                    pred_y) + " " + str(pred_z) + " " + str(pred_theta)
                points = extract_points(scene, begin - 1, box).reshape(-1, 3)
                points_name = str(begin).zfill(6) + ".npy"
                points_path = os.path.join(trajectory + "/points", points_name)
                output = net_predict(np.load(points_path).reshape(-1, 3), points)
                output = output.view(-1).cpu().detach().numpy()
                if output[0] > 0.5:
                    pred_x += output[1]
                    pred_y += output[2]
                    pred_z += output[3]
                    pred_theta += output[4]
                # Save the points
                box = "0 0 0 0 0 0 0 0 0 0 " + str(h) + " " + str(w) + " " + str(l) + " " + str(pred_x) + " " + str(
                    pred_y) + " " + str(pred_z) + " " + str(pred_theta)
                points = extract_points(scene, begin - 1, box)
                points_name = str(begin - 1).zfill(6) + ".npy"
                points_path = os.path.join(trajectory + "/points", points_name)
                np.save(points_path, points)
                # Write the predicted position back to the label file
                label_name = str(begin - 1).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "/labels", label_name)
                with open(label_path, "w") as f:
                    f.write(
                        str(pred_x) + " " + str(pred_y) + " " + str(pred_z) + " " + str(l) + " " + str(h) + " " + str(
                            w) + " " + str(pred_theta))
                print("Add frame " + str(begin - 1) + " to trajectory " + trajectory)
            # Predict the trajectory forward
            if end != last_frame:
                pred_x = predict.time_predict(position_list[:, 0])
                pred_y = predict.time_predict(position_list[:, 1])
                pred_z = predict.time_predict(position_list[:, 2])
                pred_theta = predict.time_predict(position_list[:, 3])
                # Extract the point cloud and invoke the network, remember add the point cloud to points
                scene = os.path.dirname(trajectory)
                box = "0 0 0 0 0 0 0 0 0 0 " + str(h) + " " + str(w) + " " + str(l) + " " + str(pred_x) + " " + str(
                    pred_y) + " " + str(pred_z) + " " + str(pred_theta)
                points = extract_points(scene, end + 1, box).reshape(-1, 3)
                points_name = str(end).zfill(6) + ".npy"
                points_path = os.path.join(trajectory + "/points", points_name)
                output = net_predict(np.load(points_path).reshape(-1, 3), points)
                output = output.view(-1).cpu().detach().numpy()
                if output[0] > 0.5:
                    pred_x += output[1]
                    pred_y += output[2]
                    pred_z += output[3]
                    pred_theta += output[4]
                # Save the points
                box = "0 0 0 0 0 0 0 0 0 0 " + str(h) + " " + str(w) + " " + str(l) + " " + str(pred_x) + " " + str(
                    pred_y) + " " + str(pred_z) + " " + str(pred_theta)
                points = extract_points(scene, end + 1, box)
                points_name = str(end + 1).zfill(6) + ".npy"
                points_path = os.path.join(trajectory + "/points", points_name)
                np.save(points_path, points)
                # Write the predicted position back to the label file
                label_name = str(end + 1).zfill(6) + ".txt"
                label_path = os.path.join(trajectory + "/labels", label_name)
                with open(label_path, "w") as f:
                    f.write(
                        str(pred_x) + " " + str(pred_y) + " " + str(pred_z) + " " + str(l) + " " + str(h) + " " + str(
                            w) + " " + str(pred_theta))
                print("Add frame " + str(end + 1) + " to trajectory " + trajectory)
    # Predict both ends of the trajectory and merge overlapping trajectories.
    print("Begin to merge overlapping trajectories")
    scene_paths = get_scene_list(data_dir)
    for scene in scene_paths:
        print("Merge trajectories in scene " + scene)
        trajectory_names = os.listdir(scene)
        trajectory_names.sort()
        trajectory_paths = [os.path.join(scene, trajectory_name) for trajectory_name in trajectory_names]
        while len(trajectory_paths) > 0:
            for trajectory in trajectory_paths:
                print("Processing trajectory " + trajectory)
                add_count = 0
                # Initialize l, h, w of the trajectory
                l, h, w = 0, 0, 0
                label_names = os.listdir(trajectory + "/labels")
                if len(label_names) == 0:
                    continue
                label_names.sort()
                # Get the frame range of the trajectory
                first_frame = int(label_names[0].split(".")[0])
                last_frame = int(label_names[-1].split(".")[0])
                # If the trajectory is too short, skip it
                if last_frame - first_frame < 3:
                    trajectory_paths.remove(trajectory)
                    print("Trajectory " + trajectory + " finished")
                    continue
                position_list = []
                # Get x, y, z, theta of the trajectory
                for i in range(first_frame, last_frame + 1):
                    label_name = str(i).zfill(6) + ".txt"
                    label_path = os.path.join(trajectory + "/labels", label_name)
                    with open(label_path, "r") as f:
                        line = f.readline()
                        data = line.split(" ")
                        position_list.append([float(data[0]), float(data[1]), float(data[2]), float(data[6])])
                        l, h, w = float(data[3]), float(data[4]), float(data[5])
                position_list = np.array(position_list)
                # Predict the trajectory backward first
                if first_frame != 0:
                    pred_x = predict.time_predict(np.flipud(position_list[:, 0]))
                    pred_y = predict.time_predict(np.flipud(position_list[:, 1]))
                    pred_z = predict.time_predict(np.flipud(position_list[:, 2]))
                    pred_theta = predict.time_predict(np.flipud(position_list[:, 3]))
                    # Extract the point cloud and invoke the network, remember add the point cloud to points
                    scene = os.path.dirname(trajectory)
                    box = "0 0 0 0 0 0 0 0 0 0 " + str(h) + " " + str(w) + " " + str(l) + " " + str(pred_x) + " " + str(
                        pred_y) + " " + str(pred_z) + " " + str(pred_theta)
                    points = extract_points(scene, first_frame - 1, box).reshape(-1, 3)
                    points_name = str(first_frame).zfill(6) + ".npy"
                    points_path = os.path.join(trajectory + "/points", points_name)
                    output = net_predict(np.load(points_path).reshape(-1, 3), points)
                    output = output.view(-1).cpu().detach().numpy()
                    if output[0] > 0.5:
                        print("Add frame " + str(first_frame - 1) + " to trajectory " + trajectory)
                        add_count += 1
                        pred_x += output[1]
                        pred_y += output[2]
                        pred_z += output[3]
                        pred_theta += output[4]
                        # Check if the predicted trajectory is overlapping with the previous trajectory
                        label_paths = get_all_other_labels(scene, trajectory, first_frame - 1)
                        merged = False
                        for label_path in label_paths:
                            with open(label_path, "r") as f:
                                line = f.readline()
                                data = line.split(" ")
                                # If the distance between the predicted box and the previous box is less than 0.3m and theta is less than 0.3rad, merge the two trajectories
                                if (pred_x - float(data[0])) ** 2 + (pred_y - float(data[1])) ** 2 + (pred_z - float(data[2])) ** 2 < 0.09 and abs(
                                        pred_theta - float(data[6])) < 0.3:
                                    # Get the trajectory path
                                    trajectory_path = os.path.dirname(os.path.dirname(label_path))
                                    # Move files in trajectory_path/labels and trajectory_path/points to trajectory/labels and trajectory/points
                                    move_files(trajectory_path, trajectory)
                                    # Delete the trajectory path
                                    trajectory_paths.remove(trajectory_path)
                                    print("Merge trajectory " + trajectory + " with trajectory " + trajectory_path)
                                    merged = True
                                    break
                        if not merged:
                            # Write the predicted position back to the label file
                            label_name = str(first_frame - 1).zfill(6) + ".txt"
                            label_path = os.path.join(trajectory + "/labels", label_name)
                            with open(label_path, "w") as f:
                                f.write(
                                    str(pred_x) + " " + str(pred_y) + " " + str(pred_z) + " " + str(l) + " " + str(
                                        h) + " " + str(
                                        w) + " " + str(pred_theta))
                            # Save the points
                            box = "0 0 0 0 0 0 0 0 0 0 " + str(h) + " " + str(w) + " " + str(l) + " " + str(
                                pred_x) + " " + str(
                                pred_y) + " " + str(pred_z) + " " + str(pred_theta)
                            points = extract_points(scene, first_frame - 1, box)
                            points_name = str(first_frame - 1).zfill(6) + ".npy"
                            points_path = os.path.join(trajectory + "/points", points_name)
                            np.save(points_path, points)
                # Predict the trajectory forward
                pred_x = predict.time_predict(position_list[:, 0])
                pred_y = predict.time_predict(position_list[:, 1])
                pred_z = predict.time_predict(position_list[:, 2])
                pred_theta = predict.time_predict(position_list[:, 3])
                # Extract the point cloud and invoke the network, remember add the point cloud to points
                scene = os.path.dirname(trajectory)
                box = "0 0 0 0 0 0 0 0 0 0 " + str(h) + " " + str(w) + " " + str(l) + " " + str(pred_x) + " " + str(
                    pred_y) + " " + str(pred_z) + " " + str(pred_theta)
                # If meet 'No such file or directory' error, skip this trajectory
                try:
                    points = extract_points(scene, last_frame + 1, box).reshape(-1, 3)
                except FileNotFoundError:
                    print("No such file or directory: " + scene + "/" + str(last_frame + 1))
                    trajectory_paths.remove(trajectory)
                    print("Trajectory " + trajectory + " finished")
                    continue
                points_name = str(last_frame).zfill(6) + ".npy"
                points_path = os.path.join(trajectory + "/points", points_name)
                output = net_predict(np.load(points_path).reshape(-1, 3), points)
                output = output.view(-1).cpu().detach().numpy()
                if output[0] > 0.5:
                    print("Add frame " + str(last_frame + 1) + " to trajectory " + trajectory)
                    add_count += 1
                    pred_x += output[1]
                    pred_y += output[2]
                    pred_z += output[3]
                    pred_theta += output[4]
                    # Check if the predicted trajectory is overlapping with the previous trajectory
                    label_paths = get_all_other_labels(scene, trajectory, last_frame + 1)
                    merged = False
                    for label_path in label_paths:
                        with open(label_path, "r") as f:
                            line = f.readline()
                            data = line.split(" ")
                            # If the distance between the predicted box and the previous box is less than 0.3m and theta is less than 0.3rad, merge the two trajectories
                            if (pred_x - float(data[0])) ** 2 + (pred_y - float(data[1])) ** 2 + (
                                    pred_z - float(data[2])) ** 2 < 0.09 and abs(
                                    pred_theta - float(data[6])) < 0.3:
                                # Get the trajectory path
                                trajectory_path = os.path.dirname(os.path.dirname(label_path))
                                # Move files in trajectory_path/labels and trajectory_path/points to trajectory/labels and trajectory/points
                                move_files(trajectory_path, trajectory)
                                # Delete the trajectory path
                                trajectory_paths.remove(trajectory_path)
                                print("Merge trajectory " + trajectory + " with trajectory " + trajectory_path)
                                merged = True
                                break
                    if not merged:
                        # Write the predicted position back to the label file
                        label_name = str(last_frame + 1).zfill(6) + ".txt"
                        label_path = os.path.join(trajectory + "/labels", label_name)
                        with open(label_path, "w") as f:
                            f.write(
                                str(pred_x) + " " + str(pred_y) + " " + str(pred_z) + " " + str(l) + " " + str(
                                    h) + " " + str(
                                    w) + " " + str(pred_theta))
                        # Save the points
                        box = "0 0 0 0 0 0 0 0 0 0 " + str(h) + " " + str(w) + " " + str(l) + " " + str(
                            pred_x) + " " + str(
                            pred_y) + " " + str(pred_z) + " " + str(pred_theta)
                        points = extract_points(scene, last_frame + 1, box)
                        points_name = str(last_frame + 1).zfill(6) + ".npy"
                        points_path = os.path.join(trajectory + "/points", points_name)
                        np.save(points_path, points)
                if add_count == 0:
                    trajectory_paths.remove(trajectory)
                    print("Trajectory " + trajectory + " finished")
