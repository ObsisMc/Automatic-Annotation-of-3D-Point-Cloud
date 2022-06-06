#!/bin/bash
# you should run this in the root of project, or you need to modify variable "project_root" in the following
# conda activate pcdet before run this


# data before complement
data_path=/home2/lie/InnovativePractice2/data/kitti/tracking/extracted_08_2_test
data_bck_path=/home2/lie/InnovativePractice2/data/kitti/tracking/extracted__08_2_test_bck

# intermediate data and output
object_path=/home2/lie/InnovativePractice2/AB3DMOT/results/pv_rcnn_epoch_8369_08_test/object_format_final
track_path=/home2/lie/InnovativePractice2/AB3DMOT/results/pv_rcnn_epoch_8369_08_test/tracking_format_final

# path
project_root=.
format_utils=common_utils/format_utils/
# code
complement_relative_path=predict/complement.py
e2o_relative_path=extract2object.py
o2t_relative_path=objectLabel2Tracking.py

# remove last data and cp back-up data to data
echo "Clean old data and copy back-up data"
rm -r ${data_path}
cp -r ${data_bck_path} ${data_path}
echo "Finish data preparation"

# run complement.py
echo "Begin complementing"
cd ${project_root}
python ${complement_relative_path}
echo "Finish complement"

# extract2Object2track
cd ${format_utils}

echo "Clean old object result and transfer from extract to object"
rm -r ${object_path}
mkdir ${object_path}
python ${e2o_relative_path}

echo "Clean old track result and transfer from object to track"
rm -r ${track_path}
mkdir ${track_path}
python ${o2t_relative_path}
echo "Finish all procedures :)"

