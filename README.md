# InnovativePractice1_SUSTech

This project's name is ***Joint Detection and Tracking for Automatic Annotation of 3D AD Data***. For details, please read report.
##Object detection
使用PVRCNN，数据集为kitti
> 注意：
> 1. ImageSets中的txt用来表示测试集、验证集和训练集分别使用哪些序号的数据
> 2. yaml配置文件需要修改，其 ROAD_PLANE 默认是ture.
## Visualize module
Visualize module is cloned from https://github.com/junming259/Partial_Point_Clouds_Generation and there are some changes:
#### run in .py
Argument *shrun* of functions allows you to run codes out of command lines.
#### no angle correction
Argument *oriangle* of functions allows you to extract points without angle correction and visualize it with original angle.
However, remember that if you extract points without correction, you should visualize it without correction as well.

