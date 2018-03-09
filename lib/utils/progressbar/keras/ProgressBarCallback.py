import sys

from keras import callbacks
from lib.utils import timer

class ProgressBar(callbacks.Callback):
    def __init__(self):
        super(ProgressBar, self).__init__()
        self.train_timer = timer.Timer()
        self.train_count = 0
        self.currentEpoch = 0
        self.totalEpoch = 0
        self.currentBatchShow = ""
    def on_train_begin(self, logs={}):
        print("Start training...")

        if "steps" in self.params and self.params['steps'] != None:
            self.train_count = int(self.params["steps"])
        else:
            self.train_count = int(int(self.params["samples"]) / self.params['batch_size'])
        self.totalEpoch = self.params['epochs']

    def on_train_end(self, logs={}):
        print("End training")

    def on_epoch_begin(self, epoch, logs={}):
        self.train_timer.tic()
        self.currentEpoch = epoch + 1

    def on_epoch_end(self, epoch, logs={}):
        self.train_timer.toc()
        if 'val_loss' in logs and 'val_acc' in logs:
            s1 = "\r%s,Loss=%s,Accuracy=%s,Val_Loss=%s,Val_Accuracy=%s"%(
                self.currentBatchShow,"{:.3f}".format(logs.get('loss')),"{:.3f}".format(logs.get('acc')),
                "{:.3f}".format(logs.get('val_loss')),
                                "{:.3f}".format(logs.get('val_acc')))

            sys.stdout.write(s1)
            sys.stdout.flush()

        print("\r")

    def on_batch_begin(self, batch, logs={}):
        pass
    def on_batch_end(self, batch, logs={}):
        # logs = {'batch': 26, 'size': 64, 'loss': 6.9849586, 'acc'}
        self.train_timer.remain(batch, self.train_count)
        perCount = int(self.train_count / 100) # 7
        perCount = 1 if perCount == 0 else perCount
        percent = int(batch / perCount)

        if self.train_count % perCount == 0:
            dotcount = int(self.train_count / perCount) - 1
        else:
            dotcount = int(self.train_count / perCount)

        s1 = "\rEpoch:%d / %d [%s%s] %d/%d"%(
            self.currentEpoch,
            self.totalEpoch,
            "*"*(int(percent)),
            " "*(dotcount-int(percent)),
            (batch + 1),
            (self.train_count))
        s2 = "%s,Loss=%s,Accuracy=%s,remain=%s" % (
            s1, "{:.3f}".format(logs.get('loss')),
            "{:.3f}".format(logs.get('acc')) if 'acc' in logs else "",
            self.train_timer.remain(batch, self.train_count))
        self.currentBatchShow = s1
        sys.stdout.write(s2)
        sys.stdout.flush()





