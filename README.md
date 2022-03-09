# InnovativePractice1_SUSTech

This project's name is ***Joint Detection and Tracking for Automatic Annotation of 3D AD Data***. For details, please read report.

## Visualize module
Visualize module is cloned from https://github.com/junming259/Partial_Point_Clouds_Generation and there are some changes:
#### run in .py
Argument *shrun* of functions allows you to run codes out of command lines.
#### no angle correction
Argument *oriangle* of functions allows you to extract points without angle correction and visualize it with original angle.
However, remember that if you extract points without correction, you should visualize it without correction as well.

## Usage
首先需要使用目标检测获得检测框，然后目标跟踪得到轨迹，最后采用我们的模型对结果进行进一步refine。
目标检测和目标跟踪算法的相应脚本与代码在off_the_shelf文件夹中。
### Object Detection
#### PVRCNN
数据集为kitti
> 注意：
> 1. ImageSets中的txt用来表示测试集、验证集和训练集分别使用哪些序号的数据
> 2. yaml配置文件需要修改，其 ROAD_PLANE 默认是ture.
> 3. object label中的dontcare应该总是在最后
> 4. label、image、velodyne、calib应该对应所有imagesets里的序号
> 5. 如果label里面什么也没有，生成预数据的时候会报错
> 6. tracking的某些帧可能没有label，转入openpcdet的时候要注意

`train.sh`: used to train pvrcnn model



## Work flow of the project
#### Object Detection
1. Transfer format of kitti tracking data into format of object data.


    1. 用format conversion里的track2object将tracking data导入openpcdet的data中
    2. 用off_the_shelf的createImageSets生成预数据
    3. 用openpcdet的test.py（或者off_the_shelf的test.sh）进行验证集测试
2. Generate bbox 

> tracking点云的0001文件夹176后面缺了几帧