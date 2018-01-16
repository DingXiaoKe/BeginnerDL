from keras_models.cifar import densenet
from keras_config.cifar10Config import Cifar10Config
from keras.optimizers import Adam
import os
from keras_datareaders.ClassificationReader import ClassificationReader
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import array_to_img
cfg = Cifar10Config()
densenet_depth = 40
densenet_growth_rate = 12

model = densenet(depth=densenet_depth,
                       growth_rate = densenet_growth_rate)
Model_File = "" # TODO
model.load_weights(Model_File)
optimizer = Adam()
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

labels = cfg.LABELS
reader = ClassificationReader(dataPath=os.path.join(cfg.DATAPATH, "cifar10"))
x_test, y_test = reader.readData(phrase="test")
y_test = to_categorical(y_test, num_classes=cfg.NUM_OUTPUTS)


# Evaluate model with test data set and share sample prediction results
evaluation = model.evaluate(x_test,y_test,batch_size=cfg.BATCH_SIZE,verbose=1)
print('Model Accuracy = %.2f' % (evaluation[1]))

counter = 0
figure = plt.figure()
plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9,hspace=0.5, wspace=0.3)
x_batch, y_batch = x_test[:cfg.BATCH_SIZE], y_test[:cfg.BATCH_SIZE]
predict_res = model.predict_on_batch(x_batch)
for i in range(cfg.BATCH_SIZE):
    actual_label = labels['label_names'][np.argmax(y_batch[i])]
    predicted_label = labels['label_names'][np.argmax(predict_res[i])]
    if actual_label != predicted_label:
        counter += 1
        pics_raw = x_batch[i]
        pics_raw *= 255
        pics = array_to_img(pics_raw)
        ax = plt.subplot(25//5, 5, counter)
        ax.axis('off')
        ax.set_title(predicted_label)
        plt.imshow(pics)
    if counter >= 25:
        plt.savefig("./wrong_predicted.jpg")
        break
print("Everything seems OK...")