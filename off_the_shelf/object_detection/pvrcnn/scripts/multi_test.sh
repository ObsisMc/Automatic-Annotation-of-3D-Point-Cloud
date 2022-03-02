#! /bin/bash
# cannot work
NUM_GPUS=1
path=.
CONFIG_FILE=${path}/cfgs/kitti_models/pv_rcnn.yaml
BATCH_SIZE=1
CKPT=../pv_rcnn_8369.pth
bash scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}  --save_to_file
