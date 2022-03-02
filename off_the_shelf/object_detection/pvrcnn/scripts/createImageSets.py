import os

openpcdet = "/home2/lie/InnovativePractice2/OpenPCDet"
imagesets = "data/kitti/ImageSets"
testing = "data/kitti/testing"
training = "data/kitti/training"
reference = "velodyne"
referencesuffix = ".bin"
valratio = 0.5


def getdatalist(filepath, suffix=referencesuffix):
    files = sorted(os.listdir(filepath), key=lambda x: int(x.rstrip(suffix)), reverse=False)
    return files


# for testing data
with open(os.path.join(openpcdet, imagesets, "test.txt"), "w") as f:
    f.write("")
    for file in getdatalist(os.path.join(openpcdet, testing, reference)):
        f.write(file.rstrip(referencesuffix))
        f.write("\n")

trainingdata = getdatalist(os.path.join(openpcdet, training, reference))
ntotal = len(trainingdata)
ntrain = int(ntotal * (1 - valratio))
# training data
with open(os.path.join(openpcdet, imagesets, "train.txt"), "w") as f:
    f.write("")
    for i in range(0, ntrain):
        f.write(trainingdata[i].rstrip(referencesuffix))
        f.write("\n")

# valiation data
with open(os.path.join(openpcdet, imagesets, "val.txt"), "w") as f:
    f.write("")
    for i in range(ntrain, ntotal):
        f.write(trainingdata[i].rstrip(referencesuffix))
        f.write("\n")
