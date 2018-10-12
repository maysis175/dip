import numpy as np
from mnist import MNIST
from numpy.core.multiarray import ndarray
from typing import Union

mndata = MNIST("/export/home/016/a0161419/le4nn/")

SIZEX, SIZEY = 28, 28
PIC_LEARN = 60000
PIC_TEST = 10000
M = 100              # There are M nodes on the intermediate layer
CLASS = 10


# ========================================
# Function Definition
# ========================================

# Sigmoid function (as activate function)
def sigmoid(t):
    # Avoid stack overflow
    return np.where(t <= -710, 0, (1 / (1 + np.exp(-t))))

# Softmax function (as activate function)
def softmax(a):
    alpha = a.max()
    den_y2 = 0
    for i in range(CLASS):
        den_y2 += np.exp(a[i] - alpha)
    y2 = np.exp(a - alpha) / den_y2
    return y2

# Cross entropy error (as loss function)
def cEntropy(y, y2):
    entropy = 0
    for i in range(y.size):
        entropy -= y[i] * np.log(y2[i])
    return entropy

def layer(seed, x, co_y, afun):
    np.random.seed(seed)
    W = np.random.normal(0, 1. / x.size, co_y * x.size)
    W = W.reshape(co_y, x.size)
    b = np.random.normal(0, 1. / x.size, co_y)
    b = b.reshape(co_y, 1)
    t = W.dot(x) + b
    return afun(t)

# Loss function
# Converts integer to one-hot vector and calc loss
def lossFun(y, y2):
    y_arr = np.zeros(10)
    y_arr[y] = 1
    return cEntropy(y_arr, y2)


# =========================================
# Execution Unit
# =========================================

batch = 100
if batch >= 0 and batch < PIC_LEARN:
    # Preprocessing
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0],SIZEX,SIZEY))
    Y = np.array(Y)

    #import matplotlib.pyplot as plt
    #from pylab import cm
    #plt.imshow(X[idx], cmap=cm.gray)
    #plt.show()

    # Choose batches randomly
    arr_idx = np.random.choice(PIC_LEARN, batch)

    entropy_ave = 0

    for idx in arr_idx:
        # Input layer
        # Convert the image data to a vector which has (SIZEX * SIZEY) dims
        x = X[idx].ravel()
        x = np.matrix(x).T

        y1 = layer(5, x, M, sigmoid)        # Output from intermediate layer
        y2 = layer(10, y1, CLASS, softmax)   # Output from output layer

        entropy_ave += lossFun(Y[idx], y2)

    entropy_ave = entropy_ave[0] / batch
    print entropy_ave

else:
    print ("Illegal Input!")
