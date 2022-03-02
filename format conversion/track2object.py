import numpy as np
import os
import shutil
import pickle
import random

# calib  will use image path to confirm number of files
# label_02 need to use splitpos.txt
imfolder_test = '/public_dataset/kitti/tracking/data_tracking_image_2/testing/image_02'
vefolder_test = '/public_dataset/kitti/tracking/data_tracking_velodyne/testing/velodyne'
cafolder_test = '/public_dataset/kitti/tracking/data_tracking_calib/testing/calib'
imfolder_train = '/public_dataset/kitti/tracking/data_tracking_image_2/training/image_02'
vefolder_train = '/public_dataset/kitti/tracking/data_tracking_velodyne/training/velodyne'
cafolder_train = '/public_dataset/kitti/tracking/data_tracking_calib/training/calib'
lafolder = '/public_dataset/kitti/tracking/data_tracking_label_2/training/label_02'

output = '/home2/lie/InnovativePractice2/OpenPCDet/data/kitti'
# output = '../test/object'
splittext = 'splitpos.txt'


def getfmt(generic):
    if generic == "image_2":
        return '.png'
    elif generic == "velodyne":
        return ".bin"
    elif generic == "calib":
        return ".txt"
    else:
        print("Wrong generic, please select from 'image_2', 'velodyne', 'calib'")
        return None


def maintransfer(tracking_file, num, type="training", generic="image_2"):
    fmt = getfmt(generic)
    if not fmt:
        return

    outputfolder = os.path.join(output, type, generic)
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    fn = 0
    for i in range(num):
        if generic == "calib":
            ff = os.path.join(imfolder_train if type == "training" else imfolder_test, '{:04d}'.format(i))
            cff = os.path.join(tracking_file, '{:04d}.txt'.format(i))
            fmt = ".png"
        else:
            ff = os.path.join(tracking_file, '{:04d}'.format(i))
        files = sorted(os.listdir(ff), key=lambda x: int(x.rstrip(fmt)))
        p = 0
        for file in files:
            p = int(file.rstrip(fmt))
            name = '{:06d}{}'.format(p + fn, ".txt" if generic == "calib" else fmt)
            if generic == "calib":
                fp = cff
            else:
                fp = os.path.join(ff, file)
            ofp = os.path.join(outputfolder, name)
            shutil.copy(fp, ofp)
        fn += p + 1  # record num of files
        if i == 0:
            with open(os.path.join(output, type, splittext), 'w') as f:
                f.write('')
        with open(os.path.join(output, type, splittext), 'a+') as f:
            f.write(str(fn))
            f.write('\n')
    print("Finish {} {}!".format(type, generic))


def changelabel(label):
    if label != "DontCare" and label != "Cyclist" and label != "Pedestrian":
        return "Car"
    return label


# before use this, delete previous labels.
# Must guarantee that splitpos.txt match number of labels.
def trans_label(tracking_file, num):
    targetf = 'label_2'
    type = "training"
    # read splitpos.txt
    splitposf = open(os.path.join(output, type, splittext), 'r')

    outputfolder = os.path.join(output, type, targetf)
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    checksum = 0
    for i in range(num):
        # number of labels.txt should be less than or equal line number in splitpos.txt
        labelnum = int(splitposf.readline().rstrip("\\n")) if i > 0 else 0
        ff = os.path.join(tracking_file, '{:04d}.txt'.format(i))
        with open(ff, 'r') as f:
            labels = f.readlines()
            lastframe = 0
            dontcare_stack = []
            for label in labels:
                label = label.rstrip("\\n").split(" ")
                frame, objectlabel = int(label[0]), " ".join(label[2:])
                if lastframe != frame:
                    name = "{:06d}.txt".format(lastframe + labelnum)
                    with open(os.path.join(outputfolder, name), 'a+') as f:
                        for doncare in dontcare_stack:
                            f.write(doncare)
                    dontcare_stack.clear()
                if label[1] == "-1":
                    dontcare_stack.append(objectlabel)
                    lastframe = frame
                    continue
                name = "{:06d}.txt".format(frame + labelnum)
                outlabel = open(os.path.join(outputfolder, name),
                                'a+')  # if there is nothing in a frame, a file still needs to be created
                outlabel.write(objectlabel)
                outlabel.close()
                lastframe = frame
            if dontcare_stack:
                name = "{:06d}.txt".format(lastframe + labelnum)
                with open(os.path.join(outputfolder, name), 'a+') as f:
                    for doncare in dontcare_stack:
                        f.write(doncare)
                dontcare_stack.clear()

    print("Finish training label!")
    splitposf.close()


def tmpmethod():
    for i in range(3):
        path = os.path.join(imfolder_test, "{:04d}".format(i))
        if not os.path.exists(path):
            os.makedirs(path)
        for j in range(random.randint(10, 20)):
            with open(os.path.join(path, "{:06d}.txt".format(j)), 'w') as f:
                f.write("%d" % (i * 100000 + j))


if __name__ == '__main__':
    n = 2
    data = {"testing": {"image_2": imfolder_test, "velodyne": vefolder_test, "calib": cafolder_test},
            "training": {"image_2": imfolder_train, "velodyne": vefolder_train, "calib": cafolder_train,
                         "label_2": lafolder}}
    for type in data:
        for generic in data[type]:
            if generic == "label_2":
                trans_label(data[type][generic], n)
            else:
                maintransfer(data[type][generic], n, type=type, generic=generic)
