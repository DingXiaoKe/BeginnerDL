import numpy as np


data = np.arange(7)
data = [data] * 7 * 2
data = np.array(data)

data = np.reshape(data, newshape=(2,7,7))
print(data)

data = np.transpose(data, (1,2,0))
print(data)