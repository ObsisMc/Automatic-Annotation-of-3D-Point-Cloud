#! /bin/bash
testing=~/InnovativePractice2/OpenPCDet/data/kitti/testing
training=~/InnovativePractice2/OpenPCDet/data/kitti/training
test_data_velodyne=/public_dataset/kitti/object/data_object_velodyne/testing/velodyne
test_data_image=/public_dataset/kitti/object/data_object_image_2/testing/image_2
echo "Begin import data into OpenPcdet..."
echo "Import testing data to $testing:"
echo "cp $test_data_velodyne"
cp -r $test_data_velodyne $testing
echo "Finish testing velodyne!"
echo "cp $test_data_image"
cp -r $test_data_image $testing
echo "Finish testing image!"

