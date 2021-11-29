import pickle
import os

path = '../val_result/result_track0_80.pkl'
output = 'mydividelabel/{}.txt'
if not os.path.exists('mydividelabel'):
    os.mkdir('mydividelabel')

data0 = open(path, 'rb')
data = pickle.load(data0)


def switchname(name):
    if name == 'Car':
        return 0
    elif name == 'Pedestrian':
        return 1
    elif name == 'Cyclist':
        return 2
    return -1

f = open(output.format('000000'), 'w+')
for i in range(len(data)):
    num = len(data[i].get('name'))
    int1 = int(data[i].get('frame_id'))
    for j in range(num):
        score = float(data[i].get('score')[j])
        if score <= 0.5:
            continue
        f.write(str(int1))
        f.write(' ')
        f.write(str(switchname(data[i].get('name')[j])))
        f.write(' ')
        # f.write(str(data[i].get('truncated')[j]))
        # f.write(' ')
        # f.write(str(data[i].get('occluded')[j]))
        # if data[i].get('occluded')[j] != 0:
        #     print(data[i].get('occluded')[j])
        # f.write(' ')
        for i1 in data[i].get('bbox')[j]:
            f.write(str(i1))
            f.write(' ')
        # f.write(str(data[i].get('bbox')[j]))
        f.write(str(score))
        f.write(' ')

        for idx in range(len(data[i].get('dimensions')[j])):
            f.write(str(data[i].get('dimensions')[j][(idx + 1) % 3]))
            f.write(' ')
        # f.write(str(data[i].get('dimensions')[j]))
        for i1 in data[i].get('location')[j]:
            f.write(str(i1))
            f.write(' ')
        # f.write(str(data[i].get('location')[j]))
        f.write(str(data[i].get('rotation_y')[j]))
        f.write(' ')
        f.write(str(data[i].get('alpha')[j]))
        #         # f.write(' ')
        #         # for i1 in data[i].get('boxes_lidar')[j]:
        #         #     f.write(str(i1))
        #         #     f.write(' ')
        #         # f.write(str(data[i].get('boxes_lidar')[j]))
        f.write('\n')
f.close()
