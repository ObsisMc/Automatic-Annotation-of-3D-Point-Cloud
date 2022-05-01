from DataSetTemplate import DataSetTemplate
import common_utils.cfgs as Config
import numpy as np


class SmoothTrajDataSet(DataSetTemplate):
    def __init__(self, datapath, max_traj_n=10):
        super().__init__(datapath)
        self.max_traj_n = max_traj_n
        self.points_dict = Config.load_pillar_data_template(max_traj_n)
        self.traj_list = self.datasetcreator.getWorldTrajectary(max_traj_n=self.max_traj_n)  # [[points], [labels]]

    def __getitem__(self, index):
        """
        @return [[dx,dy,d\theta], [], ...], points_dict with pillars
        """
        size = len(self.traj_list[index][0])
        # get pose diff
        traj_label_list = self.traj_list[index][1]
        poses = np.zeros((size, 3))  # [x_i - x_{i-1}, y_i - y_{i-1},  d\theta], set the first 0
        last_pose = None
        for i in range(size):
            label_path = traj_label_list[i]
            with open(label_path, "r") as f:
                label = np.array(f.readline().split(" ")).astype(np.float64)  # [x,y,z,,l,h,w,angle]
                pose = label[[0, 1, -1]]
                if last_pose is not None:
                    poses[i] = pose - last_pose
                last_pose = pose

        # get points
        traj_points_list = self.traj_list[index][0]
        for i in range(size):
            self.points_dict[i]["points"] = np.load(traj_points_list[i])

        # sample, augment and label
        poses, self.points_dict = self.sampler.paddingTraj(poses, self.points_dict, size, self.max_traj_n)
        poses, self.points_dict, labels = self.augmentor.guassianTrajAug(poses, self.points_dict,
                                                                         max_size=self.max_traj_n, actual_size=size)

        return self.points_dict, poses, labels

    def __len__(self):
        return len(self.traj_label_list)


if __name__ == "__main__":
    smooth = SmoothTrajDataSet("/home/zrh/Data/kitti/tracking/extracted_points")
    points_test, pose_test, labels = smooth[90]
    print(pose_test)
    print(labels)
