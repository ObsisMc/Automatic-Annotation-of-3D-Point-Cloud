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

#### automatic workflow
1. data preparation
   1. transport (need transport code from dataset to models)
   2. generate data info for pvrcnn ( need preprocess code for every model)
2. off_the_shelf result
   1. pvrcnn detection ( entry for testing for every detection model)
   2. transport to ab3dmot tracking 
      1. intermediate process between detection and tracking data (include abandoning low confidence)
   3. AB3DMOT get result (entry for testing for tracking model)
3. our net
   1. data preperation (from model format to our net dataset)
   2. predict proposed box ()
   3. generate our dataset 
   4. use net to adjust box (entry to test)
   5. check overlap, consolidate trajectory
   6. loop
4. post preprocess

5. sdf

## Related work
#### Point Cloud Registration
1. PCRNet
2. Deep Closest Point
3. PointNetLK