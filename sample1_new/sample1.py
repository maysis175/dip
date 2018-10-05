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

# M = 100 layers
def intermLayer(x):
    np.random.seed(5)
    W1 = np.random.random((M, SIZEX * SIZEY))
    W1 = (0.5 - W1)
    b1 = (0.5 - np.random.random((M, 1))) # type: Union[float, ndarray]
    b1 = b1.T
    t = W1.dot(x) + b1
    t = t.T
    return sigmoid(t)

# Sigmoid function (as activate function)
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

# 2nd fully connected layer
def outputLayer(y1):
    np.random.seed(10)
    W2 = np.random.random((CLASS, M))
    W2 = 0.5 - W2
    b2 = np.random.random((CLASS, 1))
    b2 = 0.5 - b2
    a = W2.dot(y1) + b2
    return a

# Softmax fanction (as activate function)
def softmax(a):
    alpha = a.max()
    den_y2 = 0
    for i in range(CLASS):
        den_y2 += np.exp(a[i] - alpha)
    y2 = np.exp(a - alpha) / den_y2
    return np.argmax(y2)

def layer(seed, x, co_y, afun):
    np.random.seed(seed)
    W = 0.5 - np.random.random((co_y, x.size))
    b = 0.5 - np.random.random((co_y, 1))
    t = W.dot(x) + b
    return afun(t)

# =========================================
# Execution Unit
# =========================================

idx = input("Please enter a number (0-9999): ")
idx = int(idx)
if idx >= 0 and idx < PIC_TEST:
    # Post processing
    X, Y = mndata.load_testing()
    X = np.array(X)
    X = X.reshape((X.shape[0],SIZEX,SIZEY))
    Y = np.array(Y)

    # Preprocessing
    # X : Images, Y : Teacher
    import matplotlib.pyplot as plt
    from pylab import cm
    # idx = 2500
    plt.imshow(X[idx], cmap=cm.gray)
    plt.show()
    print Y[idx]

    # Input layer
    # Convert the image data to a (SIZEX * SIZEY) vector
    x = X[idx].ravel()
    x = np.matrix(x).T

    # 1st fully connected layer
    # y1 = intermLayer(x)
    y1 = layer(5, x, M, sigmoid)
    a = layer(10, y1, CLASS, softmax)
    print a

else:
    print ("Illegal Input!")