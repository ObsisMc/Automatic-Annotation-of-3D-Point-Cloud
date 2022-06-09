# You should use test.py with '-save_to_file' to get txt firstly
import os

object_result = "/home2/lie/InnovativePractice2/AB3DMOT/results/pv_rcnn_epoch_8369_08_test/object_format_final"
splitpos = "/home2/lie/InnovativePractice2/OpenPCDet/data/kitti/training/splitpos.txt"
output = "/home2/lie/InnovativePractice2/AB3DMOT/results/pv_rcnn_epoch_8369_08_test/tracking_format_final"


def parserLable():
    pass


def splitPosition():
    with open(splitpos, 'r') as f:
        lines = f.read().rstrip("\n").split("\n")
        splitp = []
        for line in lines:
            if line == '':
                continue
            splitp.append(int(line))
        return splitp


def main(threshold=0.6):
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
                        if float(label[-1]) < threshold:
                            continue
                        framelabel = "{} {} {} {}".format(frame - baseframe, 0, label[0], " ".join(label[1:]))
                        f.write(framelabel + '\n')
                p += 1
        baseframe = endframe


if __name__ == "__main__":
    main()
