
import os
import time

import numpy as np
import cupy as cp

from sklearn.svm import SVR
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

import scipy.spatial.distance as scipydist

import matplotlib.pyplot as plt


def WaveletKernel(X, Y, a=10):
    
    
    # n_train = X.shape[0]
    # n_test = Y.shape[0]
    # kernel = np.zeros((n_train, n_test))
    # for j in range(n_test):
    #     delta = np.subtract(X, Y[j, :])
    #     # print(np.exp(-delta*delta).shape)
    #     kernel[:, j] = np.prod(np.cos(1.75*delta/a) * np.exp(-delta*delta)/(2*a*a), axis=1)
    # cupy 加速
    X = cp.array(X)
    Y = cp.array(Y)
    n_train = X.shape[0]
    n_test = Y.shape[0]
    kernel = cp.zeros((n_train, n_test))
    for j in range(n_test):
        delta = cp.subtract(X, Y[j, :])
        # print(np.exp(-delta*delta).shape)
        kernel[:, j] = cp.prod(cp.multiply(cp.cos(1.75*delta/a), cp.exp(-delta*delta)/(2*a*a)), axis=1)
    return cp.asnumpy(kernel)
# =============================================================================


def RBFKernel(X, Y, gamma=2):
    
    # t = time.process_time()
    # kernel1 = np.exp(-gamma * scipydist.cdist(X, Y, 'sqeuclidean'))
    # print('time', time.process_time() - t)
    
    # sklearn.metrics.pairwise.rbf_kernel 函数计算
    # t = time.process_time()
    # kernel2 = rbf_kernel(X, Y, gamma)
    # print('time', time.process_time() - t)
    
    # # cupy 加速计算
    # t = time.process_time()
    kernel = cp.array(scipydist.cdist(X, Y, 'sqeuclidean'))
    cp.multiply(-gamma, kernel, kernel)
    cp.exp(kernel, kernel)
    kernel3 = cp.asnumpy(kernel)
    # print('time', time.process_time() - t)
    
    # print(np.sum(np.abs(kernel3 - kernel2)))
    # print(np.sum(np.abs(kernel1 - kernel2)))
    return kernel3
# =============================================================================



def PolynomialKernel(X, Y, gamma=2., coef0=1., degree=2):
    
    # t = time.process_time()
    # kernel1 = (gamma*np.dot(X, Y.T) + coef0)**degree
    # print('time', time.process_time() - t)
    
    # # sklearn.metrics.pairwise.polynomial_kernel 函数计算
    # t = time.process_time()
    # kernel2 = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)
    # print('time', time.process_time() - t)
    
    # # cupy 加速计算
    # t = time.process_time()
    X = cp.array(X)
    Y = cp.array(Y)
    # kernel = cp.power(cp.add(cp.multiply(gamma, cp.dot(X, Y.T)), coef0), degree)
    kernel = cp.dot(X, Y.T)
    cp.multiply(gamma, kernel, kernel)
    cp.add(kernel, coef0, kernel)
    cp.power(kernel, degree, kernel)
    kernel3 = cp.asnumpy(kernel)
    # print('time', time.process_time() - t)
    
    # print(np.sum(np.abs(kernel3 - kernel1)))
    # print(np.sum(np.abs(kernel2 - kernel1)))
    
    return kernel3
# =============================================================================


def MultiKernelRBFP(X, Y, alpha=0.5, gamma1=2., gamma2=2., coef0=1., degree=2):
    
    # rbf 
    kernel1 = cp.array(scipydist.cdist(X, Y, 'sqeuclidean'))
    cp.multiply(-gamma1, kernel1, kernel1)
    cp.exp(kernel1, kernel1)
    cp.multiply(alpha, kernel1, kernel1)
    # poly
    X = cp.array(X)
    Y = cp.array(Y)
    kernel2 = cp.dot(X, Y.T)
    cp.multiply(gamma2, kernel2, kernel2)
    cp.add(kernel2, coef0, kernel2)
    cp.power(kernel2, degree, kernel2)
    cp.multiply(1-alpha, kernel2, kernel2)
    # multi-kernel-
    kernel = cp.asnumpy(cp.add(kernel1, kernel2))
    # 
    # X = cp.asnumpy(X)
    # Y = cp.asnumpy(Y)
    # kernel1 = alpha * RBFKernel(X, Y, gamma1) + (1-alpha) * PolynomialKernel(X, Y, gamma2, coef0, degree)
    # print(np.sum(np.abs(kernel - kernel1)))
    return kernel
# =============================================================================

def MultiKernelRBFPW(
        X, Y,
        alpha1=0.3, alpha2=0.3, alpha3=0.4,
        gamma1=2., gamma2=2., coef0=1., degree=2, a=0.5
    ):
    
    # rbf 
    kernel = cp.array(scipydist.cdist(X, Y, 'sqeuclidean'))
    cp.multiply(-gamma1, kernel, kernel)
    cp.exp(kernel, kernel)
    cp.multiply(alpha1, kernel, kernel)
    
    # poly
    X = cp.array(X)
    Y = cp.array(Y)
    kernel2 = cp.dot(X, Y.T)
    cp.multiply(gamma2, kernel2, kernel2)
    cp.add(kernel2, coef0, kernel2)
    cp.power(kernel2, degree, kernel2)
    cp.multiply(alpha2, kernel2, kernel2)
    kernel = cp.add(kernel, kernel2, kernel)
    del kernel2
    # wavelet kernel
    n_train = X.shape[0]
    n_test = Y.shape[0]
    kernel3 = cp.zeros((n_train, n_test))
    for j in range(n_test):
        delta = cp.subtract(X, Y[j, :])
        # print(np.exp(-delta*delta).shape)
        kernel3[:, j] = cp.prod(cp.multiply(cp.cos(1.75*delta/a), cp.exp(-delta*delta)/(2*a*a)), axis=1)
    cp.multiply(alpha3, kernel3, kernel3)
    # multi-kernel
    cp.add(kernel, kernel3, kernel)
    cp.divide(kernel, (alpha1+alpha2+alpha3), kernel)
    kernel = cp.asnumpy(kernel)
    del kernel3
    # 原始
    # X = cp.asnumpy(X)
    # Y = cp.asnumpy(Y)
    # kernel0 = alpha1 * RBFKernel(X, Y, gamma1) + \
    #           alpha2 * PolynomialKernel(X, Y, gamma2, coef0, degree) + \
    #           (1-alpha1-alpha2) * WaveletKernel(X, Y, a)
    # print(np.sum(np.abs(kernel - kernel0)))
    # print(1)
    return kernel
# =============================================================================