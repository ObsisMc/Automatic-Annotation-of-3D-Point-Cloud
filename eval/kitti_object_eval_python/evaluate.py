import time

import fire

import kitti_common as kitti
from eval import get_coco_eval_result, get_official_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(label_path='/home2/lie/zhangsh/InnovativePractice1_SUSTech/labels/kitti_track2object/training/label_2',
             result_path='/home2/lie/InnovativePractice2/AB3DMOT/results/pv_rcnn_epoch_8369_08_test/object_format_changeSize_tradition/training/label_2',
             label_split_file='/home2/lie/zhangsh/InnovativePractice1_SUSTech/eval/kitti_object_eval_python/ImageSets/tracking_all.txt',
             current_class=0,
             coco=False,
             score_thresh=-1):
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        result, ret_dict = get_official_eval_result(gt_annos, dt_annos, current_class)
        print("result: \n", result)
        return result, ret_dict


if __name__ == '__main__':
    fire.Fire()
