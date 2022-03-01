#! /bin/bash
# this for training pvrcnn, it must be placed in tools file
path=. # root of tools of pvrcnn project
nohupout=${path}/../output/nohups/train.out
nohup python ${path}/train.py --cfg_file ${path}/cfgs/kitti_models/pv_rcnn.yaml --pretrained_model ${path}/../pv_rcnn_8369.pth > ${nohupout} 2>&1 &
