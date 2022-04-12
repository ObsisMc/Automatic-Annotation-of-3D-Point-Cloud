#! /bin/bash
ptplr_cfg=pointpillar.yaml
ptplr_ckpt=pointpillar_7728.pth

pvrcnn_cfg=pv_rcnn.yaml
pvrcnn_ckpt=pv_rcnn_epoch_30.pth

ptrcnn_cfg=pointrcnn.yaml
ptrcnn_ckpt=pointrcnn_7870.pth

scnd_cfg=second.yaml
scnd_ckpt=second_7862.pth

vxlrcnn_cfg=voxel_rcnn_car.yaml
vxlrcnn_ckpt=voxel_rcnn_car_84.54.pth

prtA2_fr_cfg=PartA2_free.yaml
prtA2_fr_ckpt=PartA2_free_7872.pth

path=.
CONFIG_FILE=${path}/cfgs/kitti_models/${pvrcnn_cfg}
BATCH_SIZE=4
CKPT=../models/${pvrcnn_ckpt}
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}  --save_to_file
