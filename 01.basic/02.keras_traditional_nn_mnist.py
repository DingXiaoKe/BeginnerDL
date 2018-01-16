import os
import keras.utils as np_utils
from keras_callbacks.ProgressBarCallback import ProgressBarCallback
import numpy as np
from keras.layers import Conv2D, BatchNormalization,MaxPooling2D, Dense, Dropout, Flatten
from keras.models import Sequential

'''
Epoch:1 / 5 Loss=0.267,Accuracy=0.930,remain=0:00:00
Epoch:2 / 5 Loss=0.068,Accuracy=0.970,remain=0:00:00
Epoch:3 / 5 Loss=0.149,Accuracy=0.960,remain=0:00:00
Epoch:4 / 5 Loss=0.098,Accuracy=0.970,remain=0:00:00
Epoch:5 / 5 Loss=0.086,Accuracy=0.990,remain=0:00:00
End training
[0.038409899220141236, 0.98770000000000002]
'''

DATA_PATH = os.path.join("../data", "mnist.npz")
BATCH_SIZE = 100
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1
EPOCH_NUM = 5
f = np.load(DATA_PATH)
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()


X_train = x_train.reshape(x_train.shape[0],28,28,1).astype("float32")
X_test = x_test.reshape(x_test.shape[0],28,28,1).astype("float32")
# normalize the data to get better convergence
X_train /= 255
X_test /= 255
#preprocess the labels to get sparce vector of len 10
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#initialize keras model
model = Sequential()

model.add(Conv2D(filters=6,kernel_size=5, strides=1,padding="same", activation="relu", input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=10, kernel_size=5,strides=1,padding='valid', activation= "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.25))
model.add(Flatten())
model.add(Dense(units=120, activation="relu"))
model.add(Dropout(.5))
model.add(Dense(units=10, activation="softmax"))

#compile the model
proBar = ProgressBarCallback()
model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['acc'])
model.fit(X_train, Y_train, batch_size=BATCH_SIZE,epochs = EPOCH_NUM, verbose=0, callbacks=[proBar])
# get accuracy score on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

