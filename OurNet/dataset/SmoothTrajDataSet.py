from DataSetTemplate import DataSetTemplate
import numpy as np


class SmoothTrajDataSet(DataSetTemplate):
    def __init__(self, datapath, max_traj_n=10):
        super().__init__(datapath)
        self.max_traj_n = max_traj_n
        self.traj_list = self.datasetcreator.getWorldTrajectary(max_traj_n=self.max_traj_n)  # [[points], [labels]]

    def __getitem__(self, index):
        """
        @return [[dx,dy,d\theta], [], ...], points_dict with pillars
        """
        # get pose diff
        traj_label_list = self.traj_list[index][1]
        poses = [np.array([0.0, 0.0, 0.0, 0.0])]  # [x_i - x_{i-1}, y_i - y_{i-1},  d\theta], set the first 0
        last_pose = None
        for label_path in traj_label_list:
            with open(label_path, "r") as f:
                label = np.array(f.readline().split(" ")).astype(np.float64)  # [x,y,z,,l,h,w,angle]
                pose = label[[0, 1, -1]]
                if last_pose is not None:
                    poses.append(pose - last_pose)
                last_pose = pose
        poses = self.sampler.paddingTrajBoxs(poses, max_traj_n=self.max_traj_n)

        # get points
        traj_points_list = self.traj_list[index][0]
        points = []
        for ppath in traj_points_list:
            points.append(np.load(ppath))
        return points, pose

    def __len__(self):
        return len(self.traj_label_list)


if __name__ == "__main__":
    smooth = SmoothTrajDataSet("/home/zrh/Data/kitti/tracking/extracted_points")
    points_test, pose_test = smooth[0]
    print(points_test[1].shape)
