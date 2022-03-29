#! /bin/bash
# dir of openpcdet
testing=~/InnovativePractice2/OpenPCDet/data/kitti/testing
training=~/InnovativePractice2/OpenPCDet/data/kitti/training
# dir of data
test_data_calib=/public_dataset/kitti/object/data_object_calib/testing/calib
test_data_velodyne=/public_dataset/kitti/object/data_object_velodyne/testing/velodyne
test_data_image=/public_dataset/kitti/object/data_object_image_2/testing/image_2

train_data_calib=/public_dataset/kitti/object/data_object_calib/training/calib
train_data_velodyne=/public_dataset/kitti/object/data_object_velodyne/training/velodyne
train_data_image=/public_dataset/kitti/object/data_object_image_2/training/image_2
train_data_label=/public_dataset/kitti/object/data_object_label_2/training/label_2
echo "Begin import data into OpenPcdet..."
# testing
echo "Import testing calib to $testing:"
echo "cp $test_data_calib"
cp -r $test_data_calib $testing
echo "Finish testing calib!"

echo "Import testing velodyne to $testing:"
echo "cp $test_data_velodyne"
cp -r $test_data_velodyne $testing
echo "Finish testing velodyne!"

echo "Import testing image to $testing:"
echo "cp $test_data_image"
cp -r $test_data_image $testing
echo "Finish testing image!"
# training
echo "Import training calib to ${training}:"
echo "cp ${train_data_calib}"
cp -r ${train_data_calib} ${training}
echo "Finish training calib!"

echo "Import training velodyne to ${training}:"
echo "cp ${train_data_velodyne}"
cp -r ${train_data_velodyne} ${training}
echo "Finish training velodyne!"

echo "Import training image to ${training}:"
echo "cp ${train_data_image}"
cp -r ${train_data_image} ${training}
echo "Finish training image!"

echo "Import training label to ${training}:"
echo "cp ${train_data_label}"
cp -r ${train_data_label} ${training}
echo "Finish training label!"
