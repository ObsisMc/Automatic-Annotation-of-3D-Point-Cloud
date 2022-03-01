#! /bin/bash
path=.
CONFIG_FILE=${path}/cfgs/kitti_models/pv_rcnn.yaml
BATCH_SIZE=2
CKPT=../pv_rcnn_8369.pth
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}