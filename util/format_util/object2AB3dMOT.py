import os

object_result = "../test/object/training/label_2/"
splitpos = "../test/object/training/splitpos.txt"
output = "../test/tracking/label_2/"


def parserType(tp):
    if tp == 'Car':
        return 2
    elif tp == 'Pedestrian':
        return 1
    elif tp == 'Cyclist':
        return 3
    return 2  # consider other types car


def parserLable():
    pass


def splitPosition():
    try:
        with open(splitpos, 'r') as f:
            lines = f.read().split("\n")
            return [int(i) for i in lines]
    except:
        return 0


def main():
    splitp = splitPosition()
    if not splitp:
        print("Cannot find splitpos.txt")
        return
    if not os.path.exists(output):
        os.mkdir(output)

    files = sorted(os.listdir(object_result), key=lambda x: int(x.rstrip(".txt")))
    p = 0
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
                        framelabel = "{} {} {} {} {} {}".format(frame, parserType(label[0]),
                                                                " ".join(label[4:8]), label[-1],
                                                                "".join(label[8:15]), label[3])
                        f.write(framelabel + '\n')
                p += 1


if __name__ == "__main__":
    main()
