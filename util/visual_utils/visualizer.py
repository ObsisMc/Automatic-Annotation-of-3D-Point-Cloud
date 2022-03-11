from visual_modul import open3d_vis_utils as V, io_utils as io
from visual_modul.calibration import Calibration

basepath = "../../"
pointspath = basepath + "test/training/velodyne/0000/000000.bin"
boxpath = basepath + "test/training/label/000000.txt"
calibpath = basepath + "test/training/calib/0000.txt"

if __name__ == '__main__':
    calibration = Calibration(calibpath)  # used to transfer the axis to lidar's

    points = io.load_points(pointspath)
    boxes = io.load_boxes_from_object_txt(boxpath)
    boxes = calibration.bbox_rect_to_lidar(boxes)  # gain the right boxes
    V.draw_scenes(
        points=points, ref_boxes=boxes[0].reshape(1,-1)
    )
