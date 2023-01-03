import math
import os

path = './my_divide_car'
files = os.listdir(path)
output = 'my_size_car/{}.txt'
if not os.path.exists('my_size_car'):
    os.mkdir('my_size_car')

for file in files:
    f = open(path + "/" + file)
    iter_f = iter(f)
    flag = True
    count = 0
    l = 0
    w = 0
    h = 0
    tk = 1000
    tk2 = 1000
    tk3 = 1000
    for line in iter_f:
        if flag:
            flag = False
            s = line.split()
            count += 1
            l = s[12]
            w = s[11]
            h += float(s[10])
            continue
        s = line.split()

        alpha = float(s[5])
        y_r = float(s[16])
        theta = y_r - alpha
        tl = s[12]
        tw = s[11]
        h += float(s[10])
        count += 1
        acc = abs((abs(math.sin(theta)) + 1) * (abs(math.sin(alpha)) + 1))
        if abs(1 - acc) < abs(1 - tk):
            if abs(1 - acc) < abs(4 - tk2):
                tk = acc
                l = tl
        elif abs(4 - acc) < abs(4 - tk2):
            if abs(4 - acc) < abs(1 - tk):
                tk2 = acc
                l = tl
        elif abs(2 - acc) < abs(2 - tk3):
            tk3 = acc
            w = tw
    h = h / count
    f = open(path + "/" + file)
    iter_f = iter(f)
    for line in iter_f:
        line_t = line.split()
        f1 = open(output.format(line_t[1]), 'a')
        s = ''
        for i in range(len(line_t)):
            y_r_t = float(line_t[16])
            if i == 10:
                s += str(h)
            elif i == 11:
                s += str(w)
            elif i == 12:
                s += str(l)
            elif i == 13:
                tx = float(line_t[13])
                dx = float(w) * math.cos(y_r_t) - float(line_t[11]) * math.cos(y_r_t)
                s += str(tx - dx)
            elif i == 14:
                ty = float(line_t[14])
                dy = float(l) * math.sin(y_r_t) - float(line_t[12]) * math.sin(y_r_t)
                s += str(ty + dy)
            elif i == len(line_t) - 1:
                s += line_t[i]
                break
            else:
                s += line_t[i]
            s += ' '
        f1.write(s)
        f1.write('\n')
        



