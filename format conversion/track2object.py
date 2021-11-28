import numpy as np
import os
import shutil
import pickle


def trans_image(imfolder, num):
    gf = os.path.join(output, 'image_02')
    if not os.path.exists(gf):
        os.makedirs(gf)
    fn = 0
    for i in range(num):
        ff = os.path.join(imfolder, '{:04d}'.format(i))
        n = 0
        for file in os.listdir(ff):
            name = '{:06d}'.format(n + fn)
            fp = os.path.join(ff, file)
            nfp = os.path.join(ff, name)
            os.rename(fp, nfp)
            shutil.move(nfp, gf)
            fname = '{:06d}.png'.format(n + fn)
            os.rename(os.path.join(gf, name), os.path.join(gf, fname))
            n += 1
        fn += n
        if i == 0:
            with open(os.path.join(output, splittext), 'w') as f:
                f.write('')
        with open(os.path.join(output, splittext), 'a+') as f:
            f.write(str(fn))
            f.write('\n')
    print("Finish image!")


def trans_velodyne(vf, num):
    gf = os.path.join(output, 'velodyne')
    if not os.path.exists(gf):
        os.makedirs(gf)
    fn = 0
    for i in range(num):
        ff = os.path.join(vf, '{:04d}'.format(i))
        n = 0
        for file in os.listdir(ff):
            name = '{:06d}'.format(n + fn)
            fp = os.path.join(ff, file)
            nfp = os.path.join(ff, name)
            os.rename(fp, nfp)
            shutil.move(nfp, gf)
            fname = '{:06d}.bin'.format(n + fn)
            os.rename(os.path.join(gf, name), os.path.join(gf, fname))
            n += 1
        fn += n
        if i == 0:
            with open(os.path.join(output, splitve), 'w') as f:
                f.write('')
        with open(os.path.join(output, splitve), 'a+') as f:
            f.write(str(fn))
            f.write('\n')
    print("Finish velodyne!")


def trans_calib(calib, num):
    gf = os.path.join(output, 'calib')
    if not os.path.exists(gf):
        os.makedirs(gf)
    fn = 0
    for i in range(num):
        ff = os.path.join(calib, '{:04d}.txt'.format(i))
        n = 154
        for j in range(n):
            name = '{:06d}.txt'.format(j)
            shutil.copy(ff, os.path.join(gf, name))
        fn += 1
    print("Finish calib!")


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


imfolder = 'E:\kitti\data_tracking\image_02'
vefolder = 'E:\kitti\data_tracking\\velodyne'
cafolder = 'E:\kitti\data_tracking\\calib'
lafolder = 'E:\kitti\data_tracking\label_02'
output = 'E:\\kitti\\data_tracking\\trans'
splittext = 'splitpos.txt'
splitve = 'splitve.txt'

if __name__ == '__main__':
    n = 1
    # trans_image(imfolder, n)
    # trans_velodyne(vefolder, n)
    # trans_calib(cafolder, n)
    trans_label(lafolder, 1)
    # undo(imfolder)
    # trans_image()
