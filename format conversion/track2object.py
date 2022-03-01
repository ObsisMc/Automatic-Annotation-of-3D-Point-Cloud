import numpy as np
import os
import shutil
import pickle
import random

imfolder_test = '/public_dataset/kitti/tracking/data_tracking_image_2/testing/image_02'
vefolder_test = '/public_dataset/kitti/tracking/data_tracking_velodyne/testing/velodyne'
cafolder_test = '/public_dataset/kitti/tracking/data_tracking_calib/testing/calib'
imfolder_train = '/public_dataset/kitti/tracking/data_tracking_image_2/training/image_02'
vefolder_train = '/public_dataset/kitti/tracking/data_tracking_velodyne/training/velodyne'
cafolder_train = '/public_dataset/kitti/tracking/data_tracking_calib/training/calib'

lafolder = 'E:\kitti\data_tracking\label_02'
output = '/home2/lie/InnovativePractice2/OpenPCDet/data/kitti'
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


def trans_image(tracking_file, num, type="training", generic="image_2"):
    fmt = getfmt(generic)
    if not fmt:
        return

    outputfolder = os.path.join(output, type, generic)
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    fn = 0
    for i in range(num):
        if generic == "calib":
            ff = os.path.join(imfolder_train if type=="training" else imfolder_test, '{:04d}'.format(i))
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
            with open(os.path.join(output, splittext), 'w') as f:
                f.write('')
        with open(os.path.join(output, splittext), 'a+') as f:
            f.write(str(fn))
            f.write('\n')
    print("Finish {} {}!".format(type, generic))


def changelabel(label):
    if label != "DontCare" and label != "Cyclist" and label != "Pedestrian":
        return "Car"
    return label


def trans_label(labelfd, num):
    targetf = 'label_2'
    gf = os.path.join(output, targetf)
    if not os.path.exists(gf):
        os.makedirs(gf)
    for i in range(num):
        ff = os.path.join(labelfd, '{:04d}.txt'.format(i))
        with open(ff, 'r') as labelf:
            while True:
                line = labelf.readline()
                if line == '':
                    break
                frameid = -1
                label = None
                for ci in range(len(line)):
                    if line[ci] == ' ':
                        if frameid == -1:
                            frameid = int(line[:ci])
                        else:
                            pos = -1
                            for ci2 in range(ci + 1, len(line)):
                                if line[ci2] == ' ':
                                    label = changelabel(line[ci + 1:ci2])
                                    pos = ci2
                                    break
                            if label == "DontCare":
                                break
                            outputf = os.path.join(output, targetf, "{:06d}.txt".format(frameid))
                            with open(outputf, "a+") as f:
                                f.write(label + line[pos:])
                            break

    print("Finish labels!")


def tmpmethod():
    for i in range(3):
        path = os.path.join(imfolder, "{:04d}".format(i))
        if not os.path.exists(path):
            os.makedirs(path)
        for j in range(random.randint(10, 20)):
            with open(os.path.join(path, "{:06d}.txt".format(j)), 'w') as f:
                f.write("%d" % (i * 100000 + j))


if __name__ == '__main__':
    n = 2
    data = {"testing": {"image_2": imfolder_test, "velodyne": vefolder_test, "calib": cafolder_test},
            "training": {"image_2": imfolder_train, "velodyne": vefolder_train, "calib": cafolder_train}}
    for type in data:
        for generic in data[type]:
            trans_image(data[type][generic], n, type=type, generic=generic)
