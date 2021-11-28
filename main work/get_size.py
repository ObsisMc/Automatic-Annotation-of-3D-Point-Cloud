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
    l = 0
    w = 0
    h = 0
    for line in iter_f:
        if flag:
            flag = False
            s = line.split()
            l = s[12]
            w = s[10]
            h = s[11]
            continue
        s = line.split()
        tk = 0
        alpha = float(s[5])
        y_r = float(s[16])
        theta = y_r - alpha
        tl = s[12]
        tw = s[10]
        th = s[11]
        acc = abs((math.sin(theta) + 1) * (math.sin(alpha) + 1))
        if abs(1 - acc) < abs(1 - tk):
            tk = acc
            l = tl
            h = th
        else:
            w = tw
            h = th

for file in files:
    f = open(path + "/" + file)
    iter_f = iter(f)
    for line in iter_f:
        line_t = line.split()
        f1 = open(output.format(line_t[1]), 'a')
        s = ''
        for i in range(len(line_t)):
            if i == 10:
                s += str(w)
            elif i == 11:
                s += str(h)
            elif i == 12:
                s += str(l)
            elif i == len(line_t) - 1:
                s += line_t[i]
                break
            else:
                s += line_t[i]
            s += ' '
        f1.write(s)
        f1.write('\n')



