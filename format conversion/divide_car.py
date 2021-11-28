import os

path = './track_data'
files = os.listdir(path)
output = 'my_divide_car/{}.txt'
if not os.path.exists('my_divide_car'):
    os.mkdir('my_divide_car')

for file in files:
    f = open(path + "/" + file)
    iter_f = iter(f)
    for line in iter_f:
        line_t = line.split()
        f1 = open(output.format(line_t[1]), 'a')
        f1.write(line)
