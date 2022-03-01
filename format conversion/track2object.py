import numpy as np
import os
import shutil
import pickle
import random


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
            ff = os.path.join(imfolder, '{:04d}'.format(i))
            cff = os.path.join(tracking_file, '{:04d}.txt'.format(i))
        else:
            ff = os.path.join(tracking_file, '{:04d}'.format(i))
        files = sorted(os.listdir(ff), key=lambda x: int(x.rstrip(fmt)))
        p = 0
        for file in files:
            p = int(file.rstrip(fmt))
            name = '{:06d}{}'.format(p + fn, fmt)
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
    print("Finish {}!".format(generic))


# def trans_calib(tracking_file, num, type="training", generic="image_2"):
#     gf = os.path.join(output, 'calib')
#     if not os.path.exists(gf):
#         os.makedirs(gf)
#     fn = 0
#     for i in range(num):
#         ff = os.path.join(tracking_file, '{:04d}.txt'.format(i))
#         files = sorted(os.listdir(ff), key=lambda x: int(x.rstrip(".txt")))
#         p = 0
#         for file in files:
#             p = int(file.rstrip(".txt"))
#             name = '{:06d}.txt'.format(p + fn)
#             shutil.copy(ff, os.path.join(gf, name))
#         fn += p + 1
#     print("Finish calib!")


def track2object(tracking_file, num, type="training", generic="image_2"):
    pass


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


def undo(folder):
    mypath = os.path.join(output, 'image_02')
    with open(os.path.join(output, splittext), 'r') as f:
        files = os.listdir(mypath)
        begin = 0
        num = int(f.readline().rstrip('\n'))
        index = 0
        while True:
            ff = os.path.join(folder, '{:04d}'.format(index))
            if not os.path.exists(ff):
                os.mkdir(ff)
            for i in range(begin, num):
                file = files[i]
                filepath = os.path.join(mypath, file)
                newfilepath = os.path.join(mypath, "{:06d}.png".format(i))
                os.rename(filepath, newfilepath)
                shutil.move(newfilepath, ff)
            begin = num
            strnum = f.readline().rstrip('\n')
            if strnum == '':
                break
            num = int(strnum)
            index += 1
    with open(os.path.join(output, splittext), 'w') as f:
        f.write('')
    print("Finish undo!")
    pass


def tmpmethod():
    for i in range(3):
        path = os.path.join(imfolder, "{:04d}".format(i))
        if not os.path.exists(path):
            os.makedirs(path)
        for j in range(random.randint(10, 20)):
            with open(os.path.join(path, "{:06d}.txt".format(j)), 'w') as f:
                f.write("%d" % (i * 100000 + j))


imfolder = '../test/tracking/image_02/'
vefolder = 'E:\kitti\data_tracking\\velodyne'
cafolder = '../test/tracking/calib/'
lafolder = 'E:\kitti\data_tracking\label_02'
output = '../test/object'
splittext = 'splitpos.txt'
splitve = 'splitve.txt'

if __name__ == '__main__':
    n = 3
    # tmpmethod()
    trans_image(imfolder, n, type="training", generic="calib")
    # trans_velodyne(vefolder, n)
    # trans_calib(cafolder, n)
    # trans_label(lafolder, 1)
    # undo(imfolder)
    # trans_image()
