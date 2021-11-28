import pickle
import numpy as np
import os

path = '../val_result/result_track0_80.pkl'
output = 'mykitti/mydividelabel/{}.txt'
threshold = 0.5

if not os.path.exists('mykitti'):
    os.mkdir('mykitti')
if not os.path.exists('mykitti/mydividelabel'):
    os.mkdir('mykitti/mydividelabel')

data0 = open(path, 'rb')
data = pickle.load(data0)

for i in range(len(data)):
    num = len(data[i].get('name'))
    f = open(output.format(data[i].get('frame_id')), 'w+')
    for j in range(num):
        if float(data[i].get('score')[j]) < threshold:
            continue
        f.write(str(data[i].get('name')[j]))
        f.write(' ')
        f.write(str(data[i].get('truncated')[j]))
        f.write(' ')
        f.write(str(data[i].get('occluded')[j]))
        if data[i].get('occluded')[j] != 0:
            print(data[i].get('occluded')[j])
        f.write(' ')
        f.write(str(data[i].get('alpha')[j]+2*np.pi))
        f.write(' ')
        for i1 in data[i].get('bbox')[j]:
            f.write(str(i1))
            f.write(' ')
        # f.write(str(velodyne[i].get('bbox')[j]))
        for idx in range(len(data[i].get('dimensions')[j])):
            f.write(str(data[i].get('dimensions')[j][(idx + 1) % 3]))
            f.write(' ')
        # f.write(str(velodyne[i].get('dimensions')[j]))
        for i1 in data[i].get('location')[j]:
            f.write(str(i1))
            f.write(' ')
        # f.write(str(velodyne[i].get('location')[j]))
        f.write(str(data[i].get('rotation_y')[j]+2*np.pi))
        f.write(' ')
        f.write(str(data[i].get('score')[j]))
        #         # f.write(' ')
        #         # for i1 in velodyne[i].get('boxes_lidar')[j]:
        #         #     f.write(str(i1))
        #         #     f.write(' ')
        #         # f.write(str(velodyne[i].get('boxes_lidar')[j]))
        f.write('\n')
    f.close()
