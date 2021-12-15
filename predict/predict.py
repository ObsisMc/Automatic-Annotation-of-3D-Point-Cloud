import numpy as np


def calc_next_s(alpha, x):
    s = [0 for i in range(len(x))]
    # s[0] = np.sum(x[0:3]) / float(3)
    s[0] = x[0]
    for i in range(1, len(s)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]
    return s


def time_predict(x):
    alpha = 0.8
    s1 = calc_next_s(alpha, x)
    s2 = calc_next_s(alpha, s1)
    s3 = calc_next_s(alpha, s2)
    a3 = [(3 * s1[i] - 3 * s2[i] + s3[i]) for i in range(len(s3))]
    b3 = [((alpha / (2 * (1 - alpha) ** 2)) * (
            (6 - 5 * alpha) * s1[i] - 2 * (5 - 4 * alpha) * s2[i] + (4 - 3 * alpha) * s3[i])) for i in
          range(len(s3))]
    c3 = [(alpha ** 2 / (2 * (1 - alpha) ** 2) * (s1[i] - 2 * s2[i] + s3[i])) for i in range(len(s3))]
    pred = a3[-1] + b3[-1] * 1 + c3[-1] * (1 ** 2)
    return pred


if __name__ == '__main__':
    path = '0000.txt'
    x = []
    y = []
    z = []
    rotate = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_split = line.split()
            x.append(float(line_split[13]))
            y.append(float(line_split[14]))
            z.append(float(line_split[15]))
            rotate.append(float(line_split[16]))
    f.close()
    pred_x = time_predict(x)
    pred_y = time_predict(y)
    pred_z = time_predict(z)
    pred_rotate = time_predict(rotate)
