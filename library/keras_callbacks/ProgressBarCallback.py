import sys

from keras import callbacks as Kcallbacks

from keras_commons import timer as Mtimer

class ProgressBarCallback(Kcallbacks.Callback):
    def __init__(self):
        super(ProgressBarCallback, self).__init__()
        self.train_timer = Mtimer.Timer()
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

class ProgressBarGAN():
    def __init__(self, epoch_count, one_batch_count, pattern):
        self.total_count = one_batch_count
        self.current_index = 0
        self.current_epoch = 1
        self.epoch_count = epoch_count
        self.train_timer = Mtimer.Timer()
        self.pattern = pattern

    def show(self, *args):
        self.current_index += 1
        if self.current_index == 1 :
            self.train_timer.tic()

        perCount = int(self.total_count / 100) # 7
        perCount = 1 if perCount == 0 else perCount
        percent = int(self.current_index / perCount)

        if self.total_count % perCount == 0:
            dotcount = int(self.total_count / perCount)
        else:
            dotcount = int(self.total_count / perCount)

        s1 = "\rEpoch:%d / %d [%s%s] %d / %d "%(
            self.current_epoch,
            self.epoch_count,
            "*"*(int(percent)),
            " "*(dotcount-int(percent)),
            self.current_index,
            self.total_count
        )

        s2 = self.pattern % tuple([float("{:.3f}".format(x)) for x in args])

        s2 = "%s,%s,remain=%s" % (
            s1, s2, self.train_timer.remain(self.current_index, self.total_count))
        sys.stdout.write(s2)
        sys.stdout.flush()
        if self.current_index == self.total_count :
            self.train_timer.toc()
            self.current_index = 0
            print("\r")
            self.current_epoch += 1



