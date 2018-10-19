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
ETA = 0.01           # Learning rate
RHO = 0.1            # For Dropout function


# ========================================
# Function Definition
# ========================================

# Sigmoid function (as activate function)
def sigmoid(t):
    # Avoid stack overflow
    return np.asarray((1. / (1. + np.exp(-t))))

# ReLU function (as activate function)
def relu(t):
    # Avoid stack overflow
    return np.asarray(np.maximum(float(0), t))

# Softmax function (as activate function)
def softmax(a):
    alpha = a.max()
    y2 = np.exp(a - alpha) / np.sum(np.exp(a - alpha))
    return np.asarray(y2)

# Cross entropy error (as loss function)
def cEntropy(y, y2):
    entropy = 0
    for i in range(y.size):
        entropy -= y[i] * np.log(y2[i])
    return entropy[0]

# Set values of the weight with random numbers
def setWeight(seed, x, co_y, isW):
    np.random.seed(seed)
    if isW == 1:
        W = np.random.normal(0, np.sqrt(1. / x), co_y * x)
        return W.reshape(co_y, x)
    else:
        b = np.random.normal(0, np.sqrt(1. / x), co_y)
        return b.reshape(co_y, 1)

def layer(x, W, b, afun):
    t = W.dot(x) + b
    return afun(t), t

# Loss function
# Converts integer to one-hot vector and calc loss
def lossFun(y_arr, y2):
    return cEntropy(y_arr, y2)


# =========================================
# Execution Unit
# =========================================

np.set_printoptions(threshold=np.inf)

batch = 1000
EPOCH = 1000
if batch >= 0 and batch < PIC_LEARN:
    # Preprocessing
    X, Y = mndata.load_training()
    X = np.array(X)
    X = X.reshape((X.shape[0], SIZEX, SIZEY))
    Y = np.array(Y)

    # Choose batches randomly
    arr_idx = np.random.choice(PIC_LEARN, batch)

    i = 0
    for idx in arr_idx:
        if idx == arr_idx[0]:
            Xmat = X[idx].ravel()
            Xmat = np.matrix(Xmat).T
        else:
            Xmat = np.hstack((Xmat, np.matrix(X[idx].ravel()).T))

    Xmat = np.asarray(Xmat)
    Xmat = Xmat / 255.


    for ep in range(EPOCH):
        entropy_ave = 0
        i = 0

        for idx in arr_idx:
            # Input layer
            # Convert the image data to a vector which has (SIZEX * SIZEY) dims
            x = X[idx].ravel()
            x = x / 255.
            x = np.asarray(np.matrix(x).T)

            be_arr_idx = np.random.choice(M, int(np.floor(M * RHO)))
            be_arr = np.ones((M, 1))
            for j in be_arr_idx:
                be_arr[j, 0] = 0

            if idx == arr_idx[0]:
                Be = be_arr
            else:
                Be = np.hstack((Be, be_arr))

            if ep == 0:
                W1 = setWeight(5, 784, M, 1)
                b1 = setWeight(5, 784, M, 0)

                W2 = setWeight(10, M, CLASS, 1)
                b2 = setWeight(10, M, CLASS, 0)

            y1, a1 = layer(x, W1, b1, relu)  # Output from intermediate layer
            y1_be  = y1 * (Be[:, i:i+1])          # Dropout
            y2, a2 = layer(y1, W2, b2, softmax)   # Output from output layer

            y_arr = np.zeros(10)
            y_arr[Y[idx]] = 1
            entropy_ave += lossFun(y_arr, y2)

            if idx == arr_idx[0]:
                Amat1 = a1
                Amat2 = a2
                Ymat  = np.matrix(y_arr.ravel()).T
                Ymat1 = y1
                Ymat2 = y2
            else:
                Amat1 = np.hstack((Amat1, a1))
                Amat2 = np.hstack((Amat2, a2))
                Ymat  = np.hstack((Ymat,  np.matrix(y_arr.ravel()).T))
                Ymat1 = np.hstack((Ymat1, y1))
                Ymat2 = np.hstack((Ymat2, y2))

            i = i + 1

        entropy_ave = entropy_ave / batch
        print entropy_ave

        Ymat = np.asarray(Ymat)

        # Backward propagation
        # Update parameters
        En_over_a_2 = (Ymat2 - Ymat) / batch
        En_over_Y_1 = (W2.T).dot(En_over_a_2)
        En_over_W2  = En_over_a_2.dot(Ymat1.T)
        En_over_b2  = np.matrix(np.sum(En_over_a_2, axis=1)).T
        En_over_b2  = np.asarray(En_over_b2)

        W2 = W2 - ETA * En_over_W2
        b2 = b2 - ETA * En_over_b2

        En_over_Y_1 = En_over_Y_1 * Be

        # En_over_a_1 = (1. - En_over_Y_1) * En_over_Y_1                 # sigmoid
        En_over_a_1 = np.where((Amat1 > 0), En_over_Y_1, float(0))     # ReUL
        En_over_W1  = En_over_a_1.dot(Xmat.T)
        En_over_b1  = np.matrix(np.sum(En_over_a_1, axis=1)).T
        En_over_b1  = np.asarray(En_over_b1)

        W1 = W1 - ETA * En_over_W1
        b1 = b1 - ETA * En_over_b1

    np.savez("test.npz", W1, b1, W2, b2)

    # loaded_para = np.load("test.npz")
    # print(loaded_para['arr_0'].shape)
    # print(loaded_para['arr_1'].shape)
    # print(loaded_para['arr_2'].shape)
    # print(loaded_para['arr_3'].shape)

else:
    print ("Illegal Input!")
