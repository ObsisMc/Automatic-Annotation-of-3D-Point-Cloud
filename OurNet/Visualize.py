import wandb
import datetime


class Visualizer:
    def __init__(self, batches, name="Innovation 1", entity="stealdog"):
        """
        Args:
            batches:
            name: wandb里面project的名称
            entity: 用户名
        """
        wandb.init(project=name, entity=entity)  # 初始化
        wandb.config = {
            "batch_size": batches
        }  # 设置参数，没啥用
        self.table = wandb.Table(columns=["cloud", "x", "y", "z", "theta", "cls"])
        wandb.run.name = "train_on_" + (datetime.datetime.now()).strftime('%Y%m%d-%H:%M')

    def log(self, titles, values):
        """
        Args:
            titles: 图表的名字
            values: 值

        Returns:
        """
        wandb.log(dict(zip(titles, values)))

    def tablelog(self, target, pred, points=None):
        pointcloud = points
        if points is not None:
            pointcloud = wandb.Object3D(data_or_path=pointcloud)
        self.table.add_data(pointcloud, [target[0], pred[0]], [target[1], pred[1]], [target[2], pred[2]],
                            [target[3], pred[3]], [target[4], pred[4]])

    def finishtable(self, name):
        wandb.log({name: self.table})
        self.table = wandb.Table(columns=["cloud", "x", "y", "z", "theta", "cls"])
