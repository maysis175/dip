import numpy as np
from mnist import MNIST
mndata = MNIST("/export/home/016/a0161419/le4nn/")

SIZEX, SIZEY = 28, 28
PIC_LEARN = 60000
PIC_TEST = 10000
CLASS = 10

X, Y = mndata.load_training()
X = np.array(X)
X = X.reshape((X.shape[0],28,28))
Y = np.array(Y)

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from pylab import cm
idx = 100
plt.imshow(X[idx], cmap=cm.gray)
plt.show()
print Y[idx]