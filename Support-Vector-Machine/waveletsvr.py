


import numpy as np


def WaveletKernel(X, Y, a=2):
    # print(X.shape, Y.shape)
    # print(X-Y)
    
    # if X.ndim == 1:
    #     kernel = np.cos(1.75*(X-Y)/a) * np.exp(-np.dot((X-Y), (X-Y).T)/(2*a*a))
    #     kernel = np.prod(kernel)
    # elif X.ndim == 2:
    
    # 方法1
    # n_train = X.shape[0]
    # n_test = Y.shape[0]
    # kernel = np.zeros((n_train, n_test))
    # for i in range(n_train):
    #     for j in range(n_test):
    #         delta = X[i, :]-Y[j, :]
    #         # print(np.exp(-delta*delta).shape)
    #         kernel[i, j] = np.prod(np.cos(1.75*delta/a) * np.exp(-delta*delta)/(2*a*a))
    
    # 方法2
    n_train = X.shape[0]
    n_test = Y.shape[0]
    kernel = np.zeros((n_train, n_test))
    for j in range(n_test):
        delta = X - Y[j, :] 
        # print(np.exp(-delta*delta).shape)
        kernel[:, j] = np.prod(np.cos(1.75*delta/a) * np.exp(-delta*delta)/(2*a*a), axis=1)
    
    # 方法3
    # Y_temp = np.tile(Y, reps=(n_train, 1, 1))
    # print(Y_temp.shape)
    # delta = np.expand_dims(X, axis=1) - Y_temp 
    # print(delta.shape)
    # kernel = np.prod(np.cos(1.75*delta/a) * np.exp(-delta*delta)/(2*a*a), axis=2)
    # print(kernel.shape)
    return kernel
# ----------------------------------------------------------------------------

