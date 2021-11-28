import os

output = 'output\\{}'
TRAIN = "train.txt"
TEST = "test.txt"
VAL = "val.txt"


def generate(name: str, num: int):
    if not os.path.exists(output.format(num)):
        os.mkdir(output.format(num))
    with open(os.path.join(output.format(num), name), 'w') as f:
        for i in range(num):
            s = "{:06d}".format(i)
            f.write(s + "\n")


if __name__ == '__main__':
    generate(TRAIN, 748)
