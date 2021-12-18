import random
import os

for file in os.listdir('label_02'):
    data_list = []
    with open('label_02/' + file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            split_line = line.split(' ')
            data_list.append(split_line)
    for i in range(len(data_list)):
        ran_int = random.randint(0, 1)
        data_list[i].append(float(data_list[i][13]) + ((-1) ** ran_int) * random.gauss(0.25, 0.08))
        ran_int = random.randint(0, 1)
        data_list[i].append(float(data_list[i][14]) + ((-1) ** ran_int) * random.gauss(0.05, 0.02))
        ran_int = random.randint(0, 1)
        data_list[i].append(float(data_list[i][15]) + ((-1) ** ran_int) * random.gauss(0.1, 0.03))
        ran_int = random.randint(0, 1)
        data_list[i].append(float(data_list[i][16]) + ((-1) ** ran_int) * random.gauss(0.15, 0.05))
    with open('label_02_error/' + file, 'w') as f:
        for i in range(len(data_list)):
            data_list[i][16] = str(data_list[i][16]).rstrip('\n')
            f.write(str(data_list[i][0]) + ' ' + str(data_list[i][1]) + ' ' + str(data_list[i][2]) + ' ' + str(data_list[i][3]) + ' ' + str(data_list[i][4]) + ' ' + str(data_list[i][5]) + ' ' + str(data_list[i][6]) + ' ' + str(data_list[i][7]) + ' ' + str(data_list[i][8]) + ' ' + str(data_list[i][9]) + ' ' + str(data_list[i][10]) + ' ' + str(data_list[i][11]) + ' ' + str(data_list[i][12]) + ' ' + str(data_list[i][13]) + ' ' + str(data_list[i][14]) + ' ' + str(data_list[i][15]) + ' ' + str(data_list[i][16]) + ' ' + str(data_list[i][17]) + ' ' + str(data_list[i][18]) + ' ' + str(data_list[i][19]) + ' ' + str(data_list[i][20]) + '\n')
