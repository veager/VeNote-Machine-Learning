


import numpy as np


def WaveletKernel(X, Y, a=2):
    # print(X.shape, Y.shape)
    # print(X-Y)
    
    # if X.ndim == 1:
    #     kernel = np.cos(1.75*(X-Y)/a) * np.exp(-np.dot((X-Y), (X-Y).T)/(2*a*a))
    #     kernel = np.prod(kernel)
    # elif X.ndim == 2:
    
    n_train = X.shape[0]
    n_test = Y.shape[0]
    kernel = np.zeros((n_train, n_test))
    for i in range(n_train):
        for j in range(n_test):
            delta = X[i, :]-Y[j, :]
            # print(np.exp(-delta*delta).shape)
            kernel[i, j] = np.prod(np.cos(1.75*delta/a) * np.exp(-delta*delta)/(2*a*a))
    return kernel
# ----------------------------------------------------------------------------

