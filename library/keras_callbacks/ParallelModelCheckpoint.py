from keras.callbacks import ModelCheckpoint
def original_model(parallel_model):
    return parallel_model.get_layer('model_1')

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, path, monitor="val_acc"):
        super().__init__(path, monitor=monitor, save_weights_only=True)

    def set_model(self, model):
        super().set_model(original_model(model))