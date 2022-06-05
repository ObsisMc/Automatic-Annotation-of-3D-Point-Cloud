import os


extract_dir = '/home2/lie/InnovativePractice2/data/kitti/tracking/extracted_08_2_test/'
old_track_dir = '/home2/lie/InnovativePractice2/AB3DMOT/results/pv_rcnn_epoch_8369_08_test/data/'
new_track_dir = '/home2/lie/InnovativePractice2/AB3DMOT/results/pv_rcnn_epoch_8369_08_test/data_changeSize/'


def main():
    scene_list = os.listdir(extract_dir)
    for scene in scene_list:
        old_label = os.path.join(old_track_dir, scene + '.txt')
        new_label = os.path.join(new_track_dir, scene + '.txt')
        old_data = []
        with open(old_label, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                old_data.append(line)
        vehicle_list = os.listdir(extract_dir + scene)
        for vehicle in vehicle_list:
            vehicle_id = vehicle.split('#')[1]
            label_path = os.path.join(extract_dir, scene, vehicle, 'labels')
            label_list = os.listdir(label_path)
            for label in label_list:
                if label.endswith('.txt'):
                    with open(os.path.join(label_path, label), 'r') as f:
                        data = f.readline().strip().split(' ')
                        l, h, w = float(data[3]), float(data[4]), float(data[5])
                        break
            for i in range(len(old_data)):
                if old_data[i][1] == vehicle_id:
                    old_data[i][12] = str(l)
                    # old_data[i][10] = str(h)
                    old_data[i][11] = str(w)
        # create new label file and write data
        with open(new_label, 'w') as f:
            for line in old_data:
                f.write(' '.join(line) + '\n')


if __name__ == '__main__':
    main()
