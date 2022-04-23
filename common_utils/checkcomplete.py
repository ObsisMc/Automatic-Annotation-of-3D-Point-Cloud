import os
import re


def checkindex(folderp):
    suffix = '.bin'
    files = os.listdir(folderp)
    print("Total number: {}".format(len(files)))

    lostf = []
    filesi = []
    for file in files:
        filei = int(file.rstrip(suffix))
        filesi.append(filei)
        if len(filesi) > 1:
            for i in range(filesi[-2]+1, filei):
                lostf.append(i)
    print("Lost files: {}".format(lostf))


vefolder = 'E:\kitti\data_tracking\\velodyne\\0001'
if __name__ == '__main__':
    checkindex(vefolder)
