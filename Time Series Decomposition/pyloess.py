

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances, pairwise_distances_argmin


class LocallyWeightedRegression():
    
    def __init__(self, k, smooth_edge=True, robust=True, robust_n_iter=5):
        
        
        self.k = k
        
        self.smooth_edge   = smooth_edge
        self.robust        = robust
        self.robust_n_iter = 5
        
        return None
    # -------------------------------------------------------------------------
    def ComputeDistance(self, x, y):
        # 计算两个向量(样本)的距离
        assert x.ndim == 1
        assert x.shape == y.shape
        
        dist = np.sum(np.abs(x-y))
        return dist
    # -------------------------------------------------------------------------
    def ComputeDistanceMatrix(self, X, Y):
        # 计算两个数据集的距离矩阵,矩阵的每一行为一个样本
        
        # n_samples_X = X.shape[0]
        # n_samples_Y = Y.shape[0]
        # dist_matrix = np.zeros((n_samples_X, n_samples_Y))
        # for i in range(n_samples_X):
        #     for j in range(n_samples_Y):
        #         dist_matrix[i, j] = self.ComputeDistance(X[i, :], Y[j, :])
        
        # 或者直接使用sklearn中函数计算,计算速度更快
        dist_matrix = pairwise_distances(X, Y, 'manhattan')
        # print(dist_matrix-dist_matrix1)
        return dist_matrix
    # -------------------------------------------------------------------------
    def EpanechnikovKernel(self, x, kernel_width):
        
        y = np.zeros_like(x)
        
        y[np.abs(x) < 1] = 3/4 * (1 - x[np.abs(x) < 1]**2)
        
        return y
    # -------------------------------------------------------------------------
    def Bisquare(self, x):
        # x输入为残差的绝对值，x>=0
        
        y = np.zeros_like(x)
        
        y[np.abs(x)<1] = (1 - x[np.abs(x)<1]**2)**2
        
        return y
    # -------------------------------------------------------------------------
    def PredictWithoutSmooth(self, train_X, train_y, test_X, robustnest_weight=None):
        # 不使用 边界平滑对test_X的点经行评估
        
        n_samples_test = test_X.shape[0]
        
        y_pred = np.zeros(n_samples_test)
        
        dist_matrix = self.ComputeDistanceMatrix(test_X, train_X)
        # print(dist_matrix.shape)
        # 每个样本到另一个数据的按距离排序的样本索引
        min_index = dist_matrix.argsort(axis=1)
        
        # 每个y样本 对应的核宽度 kernel_width
        # kernel_width = []
        # for i in range(n_samples_test):
        #     kernel_width.append(dist_matrix[i, min_index[i, loess.k]])
        # kernel_width 另一种计算方法
        kernel_width = dist_matrix[list(range(n_samples_test)), min_index[:, self.k]]
        
        # 计算估计值
        for i in range(n_samples_test):
            # 邻近点索引
            neighbour_index = min_index[i, :self.k]
            # 权重
            weight = self.EpanechnikovKernel(dist_matrix[i, neighbour_index], kernel_width[i])
            # 邻近点实际值
            neighbour_y = train_y[neighbour_index]
            
            y_pred[i] = np.dot(weight, neighbour_y) / np.sum(weight)
            
        return y_pred
    # -------------------------------------------------------------------------
    def PredictWithinSmooth(self, train_X, train_y, test_X, robustnest_weight=None):
        # 使用 边界平滑对test_X的点经行评估
        
        n_samples_test = test_X.shape[0]
        
        y_pred = np.zeros(n_samples_test)
        
        dist_matrix = self.ComputeDistanceMatrix(test_X, train_X)
        # 每个样本到另一个数据的按距离排序的样本索引
        min_index = dist_matrix.argsort(axis=1)
        
        # 每个y样本 对应的核宽度 kernel_width
        kernel_width = dist_matrix[list(range(n_samples_test)), min_index[:, self.k]]
        
        one_vector = np.ones(self.k).reshape((-1, 1))
        
        # 计算估计值
        for i in range(n_samples_test):
            # 邻近点索引
            neighbour_index = min_index[i, :self.k]
            # 权重
            weight = self.EpanechnikovKernel(dist_matrix[i, neighbour_index], kernel_width[i])
            # 邻近点实际值
            neighbour_y = train_y[neighbour_index]
            # 邻近点x
            neighbour_x = train_X[neighbour_index, :]
            
            matrix_X = np.concatenate([one_vector, neighbour_x], axis=1)
            # print(matrix_X.shape)
            matrix_W = np.diag(weight)
            # print(matrix_W.shape)
            if not(robustnest_weight is None) and self.robust:
                # print(not(robustnest_weight is None), self.robust)
                neighbour_detla = robustnest_weight[neighbour_index]
                
                matrix_W = np.dot(np.diag(neighbour_detla), matrix_W)
                
            matrix_XTW = np.dot(matrix_X.T, matrix_W)
            # print(matrix_XTW.shape)
            matrix_B = np.dot(
                np.linalg.pinv(np.dot(matrix_XTW, matrix_X)),
                np.dot(matrix_XTW, neighbour_y)
            )
            
            matrix_E = np.hstack([[1], test_X[i, :]])
            # print(matrix_E)
            y_pred[i] = np.dot(matrix_E, matrix_B)
            
        return y_pred
    # -------------------------------------------------------------------------
    def Fit(self, train_X, train_y):
        
        self.train_X = train_X
        self.train_y = train_y
        
        if self.smooth_edge:
            y_pred = self.PredictWithinSmooth(train_X, train_y, train_X)
        elif not self.smooth_edge:
            y_pred = self.PredictWithoutSmooth(train_X, train_y, train_X)
        
        # 修正outlier的影响
        if self.robust:
            
            for i in range(self.robust_n_iter):
            
                error = np.abs(y_pred - train_y)
                
                median_error = np.median(error)
                
                robustnest_weight = self.Bisquare(error / (6*median_error))
                
                y_pred = self.PredictWithinSmooth(train_X, train_y, train_X, robustnest_weight)
                
        return y_pred
    # -------------------------------------------------------------------------
# =============================================================================


