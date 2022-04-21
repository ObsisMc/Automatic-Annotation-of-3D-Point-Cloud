# You should use test.py with '-save_to_file' to get txt firstly
import os

object_result = "/home2/lie/InnovativePractice2/OpenPCDet/output/cfgs/kitti_models/pv_rcnn/default/eval/epoch_30/val/default/final_result/data"
splitpos = "/home2/lie/InnovativePractice2/OpenPCDet/data/kitti/training/splitpos.txt"
output = "/home2/lie/InnovativePractice2/OpenPCDet/data/kitti/training/tracking_label"


def parserType(tp):
    if tp == 'Car':
        return 2
    elif tp == 'Pedestrian':
        return 1
    elif tp == 'Cyclist':
        return 3
    return 2  # consider other types car


def parserThreshold(tp):
    if tp == 'Car':
        return 0.85
    elif tp == 'Pedestrian':
        return 0.7
    elif tp == 'Cyclist':
        return 0.7
    return 0.8  # consider other types car


def splitPosition():
    with open(splitpos, 'r') as f:
        lines = f.read().rstrip("\n").split("\n")
        splitp = []
        for line in lines:
            if line == '':
                continue
            splitp.append(int(line))
        return splitp


def main():
    splitp = splitPosition()
    if not splitp:
        return
    if not os.path.exists(output):
        os.mkdir(output)

    files = sorted(os.listdir(object_result), key=lambda x: int(x.rstrip(".txt")))
    p = 0
    baseframe = 0
    for scene in range(len(splitp)):
        endframe = splitp[scene]
        with open(os.path.join(output, "{:04d}.txt".format(scene)), 'w') as f:
            while p < len(files):
                frame = int(files[p].rstrip(".txt"))
                if frame >= endframe:
                    break
                with open(os.path.join(object_result, files[p]), 'r') as labeltxt:
                    labels = labeltxt.read().split("\n")
                    for label in labels:
                        # data clean
                        if label == '':
                            continue
                        label = label.split(" ")
                        # filter low confidence
                        if float(label[-1]) < parserThreshold(label[0]):
                            continue
                        framelabel = "{} {} {} {} {} {}".format(frame - baseframe, parserType(label[0]),
                                                                " ".join(label[4:8]), label[-1],
                                                                " ".join(label[8:15]), label[3])
                        f.write(framelabel + '\n')
                p += 1
        baseframe = endframe


if __name__ == "__main__":
    main()
