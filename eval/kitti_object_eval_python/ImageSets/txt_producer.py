import os


# read all txt file name
result_txt_path = '/home2/lie/zhangsh/InnovativePractice1_SUSTech/labels/kitti_track2object/training/label_2/'
file_name = 'tracking_all.txt'
txt_list = os.listdir(result_txt_path)
txt_list.sort()
with open(file_name, 'w') as f:
    for txt_name in txt_list:
        # remove .txt
        number = txt_name.split('.')[0]
        f.write(number + '\n')

