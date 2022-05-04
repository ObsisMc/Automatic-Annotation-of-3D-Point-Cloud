from DataSetTemplate import DataSetTemplate
import common_utils.cfgs as Config
from dataset_utils.processor.data_processor import DataProcessor
import numpy as np

# FIXME: data has problem!!!! the world coordinate may wrong!!! use visualization to check!
class SmoothTrajDataSet(DataSetTemplate):
    """
    1. batch can only be 1
    2. only support image (70,70)
    """

    def __init__(self, datapath, max_traj_n=10):
        super().__init__(datapath)
        self.max_traj_n = max_traj_n
        self.xyz_offset = Config.load_pillar_vfe()["POINT_CLOUD_RANGE"].reshape(2, -1)
        # self.points_dict = Config.load_pillar_data_template(max_traj_n)
        self.traj_list = self.datasetcreator.getWorldTrajectary(max_traj_n=self.max_traj_n)  # [[points], [labels]]

        self.dataprocessor = DataProcessor(Config.load_pillar_vfe())

    def __getitem__(self, index):
        """
        @return [[dx,dy,d\theta], [], ...], points_dict with pillars
        """
        # index = 1076
        size = len(self.traj_list[index][0])
        # get pose diff
        traj_label_list = self.traj_list[index][1]
        poses = np.zeros((self.max_traj_n, 3))  # [x_i - x_{i-1}, y_i - y_{i-1},  d\theta], set the first 0
        centers = np.zeros((self.max_traj_n, 3))
        last_pose = None
        for i in range(size):
            label_path = traj_label_list[i]
            with open(label_path, "r") as f:
                label = np.array(f.readline().split(" ")).astype(np.float64)  # [x,y,z,,l,h,w,angle]
                pose = label[[0, 1, -1]]
                centers[i] = label[[0, 1, 2]]
                if last_pose is not None:
                    poses[i] = pose - last_pose
                last_pose = pose
        # padding center
        for i in range(size, self.max_traj_n):
            centers[i] = centers[size - 1]

        # get points
        points_dicts = Config.load_pillar_data_template(self.max_traj_n)
        traj_points_list = self.traj_list[index][0]
        for i in range(size):
            points_dicts[i]["points"] = np.load(traj_points_list[i])

        # sample, augment and label
        # if there is no frame, pose is [0,0,0], center and point is the same as the last one
        poses, points_dicts = self.sampler.paddingTraj(poses, points_dicts, size, self.max_traj_n)
        poses, points_dicts, labels = self.augmentor.guassianTrajAug(poses, points_dicts, centers,
                                                                     max_size=self.max_traj_n, actual_size=size)
        # get pillar
        for i in range(self.max_traj_n):
            point_dict = points_dicts[i]
            vrange = (self.xyz_offset + centers[i].reshape(1, -1)).reshape(-1, )
            # if point_dict.get("point_cloud_range") is not None:
            #     raise AssertionError("Point_cloud_range is not None at index %d-%d" % (index, i))

            point_dict["point_cloud_range"] = vrange
            self.dataprocessor.transform_points_to_voxels(point_dict, coors_range_xyz=vrange)
            # print(point_dict["voxels"].shape)
            if point_dict.get("voxels").shape[0] == 0:
                print(index, i)
                assert False
        return points_dicts, poses, labels  # list, np.ndarray, np.ndarray

    def __len__(self):
        return len(self.traj_list)


class SmoothTrajTestDataSet(SmoothTrajDataSet):
    def __init__(self, datapath, max_traj_n=10):
        super().__init__(datapath, max_traj_n)

    def __getitem__(self, item):
        points_dicts, poses, labels = super().__getitem__(item)

        traj_label_list = self.traj_list[item][1]
        info = {"scene": None, "tid": None, "frame": []}
        for path in traj_label_list:
            path_list = path.split("/")
            scene, tid, frame = path_list[-4], path_list[-3], path_list[-1].rstrip(".txt")
            if info["scene"] is None:
                info["scene"] = scene
            if info["tid"] is None:
                info["tid"] = tid
            info["frame"].append(frame)

        return points_dicts, poses, labels, info

    def __len__(self):
        return super().__len__()


if __name__ == "__main__":
    smooth = SmoothTrajDataSet("/home/zrh/Data/kitti/tracking/extracted_points")
    points_test, pose_test, labels = smooth[90]
    print(pose_test)
    print(labels)
