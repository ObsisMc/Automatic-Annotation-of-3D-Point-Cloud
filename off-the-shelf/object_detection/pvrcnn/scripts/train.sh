#! /bin/bash
# this for training pvrcnn
path='/home2/lie/InnovativePractice2/OpenPCDet/tools' # root of pvrcnn project
python train.py --cfg_file ${path}/cfgs/kitti_models/pv_rcnn.yaml --pretrained_model ${path}/../pv_rcnn_8369.pth