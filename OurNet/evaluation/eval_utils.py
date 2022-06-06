import logging
import os
import datetime


class EvalLog:
    def __init__(self, logdir, net):
        self.log = logging.getLogger()
        self.log.setLevel(logging.DEBUG)
        self.logdir = logdir

        self.formatter = logging.Formatter("%(asctime)s %(message)s")
        self.init_log(net)

    def init_log(self, net):
        # initial file
        date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        file_path = os.path.join(self.logdir, net.__class__.__name__, "log%s.dir" % date)
        assert not os.path.exists(file_path)

        if not os.path.exists(os.path.split(file_path)[0]):
            os.makedirs(os.path.split(file_path)[0])

        # set config
        fh = logging.FileHandler(file_path, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self.log.addHandler(fh)

    def log_eval_ckpt(self, msg):
        self.log.info(msg)
