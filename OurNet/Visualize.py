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
        wandb.init(project=name, entity=entity)
        wandb.config = {
            "batch_size": batches
        }
        wandb.run.name = "train_on_" + (datetime.datetime.now()).strftime('%Y%m%d-%H:%M')

    def log(self, titles, values):
        """
        Args:
            titles: 图表的名字
            values: 值

        Returns:
        """
        wandb.log(dict(zip(titles, values)))
