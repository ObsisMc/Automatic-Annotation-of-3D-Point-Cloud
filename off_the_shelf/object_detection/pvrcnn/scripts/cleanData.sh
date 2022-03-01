#! /bin/bash
# rm all testing and training data in openpcdet
path=/home2/lie/InnovativePractice2/OpenPCDet/data/kitti/
rm -r ${path}/testing
rm -r ${path}/training
mkdir ${path}/testing
mkdir ${path}/training
