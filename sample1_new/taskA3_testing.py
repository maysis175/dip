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

RHO = 0.1            # For Dropout


# ========================================
# Function Definition
# ========================================

# Sigmoid function (as activate function)
def sigmoid(t):
    # Avoid stack overflow
    return (1. / (1 + np.exp(-t)))

# ReLU function (as activate function)
def relu(t):
    # Avoid stack overflow
    return np.maximum(0., t)

# Softmax function (as activate function)
def softmax(a):
    alpha = a.max()
    y2 = np.exp(a - alpha) / np.sum(np.exp(a - alpha))
    return np.argmax(y2)

def layer(x, W, b, afun):
    t = W.dot(x) + b
    return afun(t)


# =========================================
# Execution Unit
# =========================================

rate = 0

idx = input("Please enter a number (0-9999): ")
idx = int(idx)
if idx >= 0 and idx < PIC_TEST:
    # Preprocessing
    X, Y = mndata.load_testing()
    X = np.array(X)
    X = X.reshape((X.shape[0], SIZEX, SIZEY))
    Y = np.array(Y)

    for i in range(idx):
        #import matplotlib.pyplot as plt
        #from pylab import cm
        #plt.imshow(X[i], cmap=cm.gray)
        #plt.show()

        # Input layer
        # Convert the image data to a vector which has (SIZEX * SIZEY) dims
        x = X[i].ravel()
        x = x / 255.
        x = np.asarray(np.matrix(x).T)

        loaded_para = np.load("test.npz")
        W1 = loaded_para['arr_0']
        b1 = loaded_para['arr_1']
        W2 = loaded_para['arr_2']
        b2 = loaded_para['arr_3']

        y1 = layer(x, W1, b1, relu)        # Output from intermediate layer
        # y1 = y1 * (1. - RHO)               # Dropout
        a = layer(y1, W2, b2, softmax)   # Output from output layer
        print Y[i], a

        if Y[i] == a:
            rate = rate + 1

    print "accuracy rate : ", (float(rate) / idx)

else:
    print ("Illegal Input!")