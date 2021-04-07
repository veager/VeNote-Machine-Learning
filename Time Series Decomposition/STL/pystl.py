

import random

import pandas as pd

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

# from sklearn.metrics import pairwise_distances, pairwise_distances_argmin



class FlexMatrix():
    # 灵活的二维数据
    def __init__(self, sequence, row_num):
        # 
        self.sequence = sequence # list
        self.row_num  = row_num  # num, 行数,周期长度
        
        self.n_sample = len(sequence)
        
        # 计算最后一个样本的行列索引
        self.ComputeIndex()
        
        self.matrix = []
        # 外循环每一行
        for jx in range(row_num):
            self.matrix.append(sequence[jx::row_num])
            
        return None
    # -------------------------------------------------------------------------
    def ComputeIndex(self):
        
         # 完整的列的个数, 周期个数
        self.col_ix = self.n_sample // self.row_num
        
        if self.n_sample > (self.col_ix)*self.row_num:
            self.row_ix = self.n_sample - (self.col_ix)*self.row_num - 1
        # 刚好是一个完整的矩阵
        else:
            self.col_ix = self.col_ix - 1
            self.row_ix = self.row_num - 1
        # print(self.row_ix, self.col_ix)
        return None
    # ------------------------------------------------------------------------
    def Append(self, element):
        
        self.sequence.append(element)
        
        self.n_sample = len(self.sequence)
        
        self.ComputeIndex()
        
        self.matrix[self.row_ix].append(element)
        
        return None
    # ------------------------------------------------------------------------
# ============================================================================


def RecombineSubseriers(all_subseriers):
    # list 嵌套 list 类型
    row_num  = len(all_subseriers)
    
    n_period = len(all_subseriers[row_num-1])
    recom_series = []
    
    for j in range(n_period):
        for i in range(row_num):
            recom_series.append(all_subseriers[i][j])
    
    for subseriers in all_subseriers:
        if len(subseriers) > n_period:
            recom_series.append(subseriers[n_period])
    return recom_series
# ============================================================================


def MoveAverage(series, order):
    # 移动平均
    assert series.ndim == 1
    return np.array([np.nanmean(series[i: i+order]) for i in range(series.shape[0]-order+1)])
# =============================================================================


def ComputeDistanceMatrix(x1, x2):
    # 计算两个序列（1-D向量）的距离矩阵
    # dist_matrix = pairwise_distances(X, Y, 'manhattan')
    dist_matrix = np.abs(x1.reshape((-1, 1)) - x2.reshape((1, -1)))
    return dist_matrix
# =============================================================================



def Tricube(x):
    # x>=0 恒成立
    
    y = np.zeros_like(x)
    
    y[x<1] = (1 - x[x<1]**3)**3
    
    return y
# =============================================================================
def Bisquare(x):
    # x输入为残差的绝对值，x>=0
    
    y = np.zeros_like(x)
    
    y[x<1] = (1 - x[x<1]**2)**2
    
    return y
# =============================================================================




def Loess(train_x, train_y, test_x, k, robust_weight):
    # train_x, train_y, test_y 均为 1-d np.array
    
    # 预处理 train_y 中的缺失值, 缺失值不能太多
    # 非nan index
    index = np.logical_not(np.isnan(train_y))
    train_x = train_x[index]
    train_y = train_y[index]
    robust_weight = robust_weight[index]
    
    n_samples_test = test_x.shape[0]
    
    y_pred = np.zeros(n_samples_test)
    
    dist_matrix = ComputeDistanceMatrix(test_x, train_x)
    # 每个样本到另一个数据的按距离排序的样本索引
    min_index = dist_matrix.argsort(axis=1)
    
    # 每个y样本 对应的核宽度 kernel_width
    kernel_width = dist_matrix[list(range(n_samples_test)), min_index[:, k]]
    
    one_vector = np.ones(k).reshape((-1, 1))
    
    # 计算估计值
    for i in range(n_samples_test):
        # 邻近点索引
        neighbour_index = min_index[i, :k]
        # 核
        kernel = dist_matrix[i, neighbour_index] / kernel_width[i]
        # 权重
        weight = Tricube(kernel)
        # 邻近点实际值
        neighbour_y = train_y[neighbour_index]
        # 邻近点x
        neighbour_x = train_x[neighbour_index].reshape((-1, 1))
        # 邻近点的 robust_weight
        neighbour_robust_weight = robust_weight[neighbour_index]
        # print(neighbour_robust_weight.shape)
        matrix_X = np.concatenate([one_vector, neighbour_x], axis=1)
        # print(matrix_X.shape)
        matrix_W = np.diag(weight)
        # print(matrix_W.shape)
        matrix_W = np.dot(np.diag(neighbour_robust_weight), matrix_W)
        
        matrix_XTW = np.dot(matrix_X.T, matrix_W)
        # print(matrix_XTW.shape)
        matrix_B = np.dot(
            np.linalg.pinv(np.dot(matrix_XTW, matrix_X)),
            np.dot(matrix_XTW, neighbour_y)
        )
        
        matrix_E = [1, test_x[i]]
        # print(matrix_E)
        y_pred[i] = np.dot(matrix_E, matrix_B)
        
    return y_pred
# =============================================================================


class SeasonalTrendDecomposition():
    
    def __init__(self, period_length, k1, k2, k3, n_inner, n_outer):
        
        self.period_length = period_length
        
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        
        self.n_inner = n_inner
        self.n_outer = n_outer
        
        return None
    # -------------------------------------------------------------------------
    def PeriodNormalization(self, periodic):
        # 周期标准化
        periodic_matrix = periodic.reshape((self.period, -1), order='F')
        one_periodic = np.mean(periodic_matrix, axis=1)
        periodic = np.tile(one_periodic, self.n_period) # repeat 按元素复制
        return periodic
    # -------------------------------------------------------------------------
    def SubSeriesLoess(self, series, robust_weight):
        # 周期子序列平滑
        
        # 提取周期子序列
        mat_series = FlexMatrix(series.tolist(), self.period_length)
        all_subseriers = mat_series.matrix
        
        mat_robust_weight = FlexMatrix(robust_weight.tolist(), self.period_length)
        all_robust_weight = mat_robust_weight.matrix
        
        all_subseriers_loess = []
        
        for ix, subseriers in enumerate(all_subseriers):
            
            subseriers_len = len(subseriers)
            # x
            x = np.array(range(1, 1+subseriers_len, 1))
            # y
            subseriers = np.array(subseriers)
            # robust_weight
            robust_weight = np.array(all_robust_weight[ix])
            
            # 待预测序列
            x1 = np.zeros(subseriers_len+2)
            x1[1:-1] = x
            x1[-1]   = 1+subseriers_len
            # print(x.shape, subseriers.shape, x1.shape, robust_weight.shape)
            
            pred_y = Loess(
                x, 
                subseriers, 
                x1, 
                self.k1, 
                robust_weight
            )
            
            all_subseriers_loess.append(pred_y.tolist())
            
        return all_subseriers_loess
    # -------------------------------------------------------------------------
    def Fit(self, series):
        
        # 样本个数
        self.n_sample = series.shape[0]
        # 原始序列
        original = series
        
        # 初始化 trend
        trend = np.zeros(self.n_sample)
        # 初始化 robust_weight
        robust_weight = np.ones(self.n_sample)
        
        for i_outer in range(self.n_outer):
            
            for i_inner in range(self.n_inner):
                # STEP 1:
                detrend = original - trend
                # STEP 2:
                # 周期子序列平滑
                all_subseriers_loess = self.SubSeriesLoess(detrend, robust_weight)
                
                temp_periodic = np.array(RecombineSubseriers(all_subseriers_loess))
                # print(temp_periodic.shape)
                # STEP 3:
                lowpass_filtering = MoveAverage(temp_periodic,     order=self.period_length)
                lowpass_filtering = MoveAverage(lowpass_filtering, order=self.period_length)
                lowpass_filtering = MoveAverage(lowpass_filtering, order=3)
                
                lowpass_filtering = Loess(
                    np.array(range(self.n_sample)), 
                    lowpass_filtering, 
                    np.array(range(self.n_sample)),
                    self.k2,
                    robust_weight
                )
                # STEP 4:
                periodic = temp_periodic[self.period_length: -self.period_length] - lowpass_filtering
                # periodic = self.PeriodicNormaloization(periodic)
                # STEP 5:
                deperiodic = original - periodic
                # STEP 6:
                trend = Loess(
                    np.array(range(self.n_sample)), 
                    deperiodic, 
                    np.array(range(self.n_sample)),
                    self.k3,
                    robust_weight
                )
                # print(trend.shape)
            # 余项
            remainder = original - periodic - trend
            
            median = np.nanmedian(np.abs(remainder))
            # 非nan索引
            index = np.logical_not(np.isnan(remainder))
            
            robust_weight[index] = np.abs(remainder[index]) / (6*median)
            
            robust_weight[index] = Bisquare(robust_weight[index])
            
            
        self.original  = original.tolist()
        self.trend     = trend.tolist()
        self.periodic  = periodic.tolist()
        self.remainder = remainder.tolist()
        self.robust_weight = robust_weight.tolist()
        # # 去周期曲线
        # self.deperiodic = deperiodic.tolist()
        # # 一个周期
        # self.one_periodic = periodic[:period]
        return trend, periodic, remainder
    # -------------------------------------------------------------------------
    def CombineResutls(self):
        
        results = pd.DataFrame(columns=['original', 'trend', 'periodic', 'remainder'])
        results['original']  = self.original
        results['trend']     = self.trend
        results['periodic']  = self.periodic
        results['remainder'] = self.remainder
        return results
    # -------------------------------------------------------------------------
# =============================================================================




# stl = SeasonalTrendDecomposition(period_length=288, loess_k=150)

# f = plt.figure(figsize=(20,6))


# x = np.linspace(0, 5, 100)
# y1 = stl.Tricube(x)
# y2 = stl.Bisquare(x)
# plt.plot(x, y1)
# plt.plot(x, y2)


# x1 = np.array([1, 2, 3])
# x2 = np.array([4, 5])
# dist_matrix = stl.ComputeDistanceMatrix(x1, x2)


# x = np.linspace(-10, 10, 1000)
# y = np.sin(x) + np.random.random(1000) * 0.5

# plt.scatter(x, y, s=5)
# y_pred = stl.Loess(x, y, x)
# plt.plot(x1, y_pred)


# x1 = np.linspace(-11, 11, 1200)
# plt.scatter(x, y, s=5)
# y_pred = stl.Loess(x, y, x1)
# plt.plot(x1, y_pred)

# index = list(range(500))
# index.extend(list(range(550, 1000)))
# x1 = x[index]
# y1 = y[index]
# plt.scatter(x1, y1, s=5)
# y_pred = stl.Loess(x1, y1, x)
# plt.plot(x, y_pred)


# x = np.arange(1, 20, 2) + np.random.random(10) * 10
# y = stl.MoveAverage(x, 4)


# series = np.arange(0, 100)

# s = FlexMatrix(series.tolist(), 30)

# matrix = s.matrix
# print(s.row_ix, s.col_ix)
# print(matrix[s.row_ix][s.col_ix])

# # for i in range(100, 200):
# #     s.Append(i)
#     # print(s.row_ix, s.col_ix)
#     # print(matrix[s.row_ix][s.col_ix])

# s1 = RecombineSubseriers(matrix)



