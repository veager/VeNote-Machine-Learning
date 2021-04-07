'''

'''

import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, RegressorMixin


class LOESS(BaseEstimator, RegressorMixin):
    '''
    sklearn 的 LOESS 回归
    '''
    def __init__(self, k=5, kernel='bisquare', distance='manhattan', p='1', istimeseries=True):
        self.k = k #
        self.kernel = kernel
        self.distance = distance
        self.p = p
        
        self.istimeseries = istimeseries
        if self.istimeseries:
            # front_length 和 behind_length 表示当前值前邻近点数量和后邻近点数量
            self.front_length  = int((self.k-1)/2)
            self.behind_length = self.k - self.front_length
            # k 与 front_length, behind_length 之间的关系
            # 当 k 为奇数时，front_length = behind_length
            # 当 k 为偶数时，front_length = behind_length - 1
            
            # 索引分界点
            #                    0 <= index < front_length 
            #         front_length <= index < x.size-behind_length
            # x.size-behind_length <= index < x.size
            # 等间距的时间序列数据
        
        # 其他的一些用于判断避对输入判断避免出错的参数
        self.istrained = False
        return None
    # --------------------------------------------
    
    def __judge_input_dims(self, X):
        if self.istimeseries:
            try:
                assert X.ndim == 1
            except:
                print('使用时间序列方法，请确保输入数据X为1维的形式！而实际X.shape={0}'.format(X.shape))
                return None
        else:
            try:
                assert X.ndim >= 2
            except:
                print('使用通用方法，请确保输入数据X为大于等于2维的形式！而实际X.shape={0}'.format(X.shape))
                return None
        return None
    # --------------------------------------------
    
    def __kernelFunction(self, x):
        '''
        kernel 函数
        '''
        if self.kernel == 'bisquare':
            arr = np.clip(x, -1, 1)
            arr = (1 - arr ** 2) ** 2
        
        elif self.kernel == 'tricube':
            arr = np.clip(x, -1, 1)
            arr = (1 - np.abs(arr) ** 3) ** 3
        
        elif self.kernel == 'epanechnikov':
            arr = np.clip(x, -1, 1)
            arr = 0.75 * (1 - np.abs(arr) ** 2)
        return arr
    # -------------------------------------------- 
    
#     def __findNeighbor_timeseries(self, x, y):
#         '''
#         用于时间序列的查找邻近样本
#         '''
#         all_near_indices = []
        
#         # 前边界索引
#         x_near_indices = [i for i in range(self.k)]
#         for t in range(self.front_length):
#             all_near_indices.append(x_near_indices)
            
#         # 中间 无边界
#         for t in range(self.front_length, x.size - self.behind_length):
#             x_near_indices = [t - self.front_length + i for i in range(self.k)]
#             all_near_indices.append(x_near_indices)
        
#         # 后边界索引
#         x_near_indices = [i for i in range(x.size-self.k, x.size)]
#         for t in range(x.size - self.behind_length, x.size):
#             all_near_indices.append(x_near_indices)
        
#         return all_near_indices
# #         return np.array(np.hstack([x.reshape(-1,1), all_near_indices]))
#     # --------------------------------------------   
    
    def __timeseries_findNeighbor_index(self, x_0):
        '''
        用于时间序列的查找邻近样本
        '''
        if x_0 < self.bound_up:
            self.x_0_position = 0
            x_0_neighbor_indices = [i for i in range(self.k)]
        elif x_0 < self.bound_low:
            self.x_0_position = 1
            x_0_index = list(self.train_x).index(x_0)
            x_0_neighbor_indices = [x_0_index - self.front_length + i for i in range(self.k)]
        else:
            self.x_0_position = 2
            x_0_neighbor_indices = [i for i in range(self.train_length - self.k, self.train_length)]
        
        return x_0_neighbor_indices
    # --------------------------------------------   
    
    def __timeseries_fit(self, x, y):
        '''
        用于判读分界点
        '''
        self.train_x = x
        self.train_y = y
        
        self.train_length = x.size
        
        # 索引分界点
        #                    0 <= index < front_length 
        #         front_length <= index < x.size-behind_length
        # x.size-behind_length <= index < x.size
        self.bound_up  = x[self.front_length]            # < self.bound_up   表示上边界点
        self.bound_low = x[x.size - self.behind_length]  # >= self.bound_low 表示下边界点
    
        return None
    # --------------------------------------------   

    def __timeseries_fit(self, x, y):
        '''
        用于判断分界点
        '''
        self.train_x = x
        self.train_y = y
        
        self.train_length = x.size
        
        # 索引分界点
        #                    0 <= index < front_length 
        #         front_length <= index < x.size-behind_length
        # x.size-behind_length <= index < x.size
        self.bound_up  = x[self.front_length]            # < self.bound_up   表示上边界点
        self.bound_low = x[x.size - self.behind_length]  # >= self.bound_low 表示下边界点
    
        return None
    # --------------------------------------------
    
    def __general_fit(self, X, y):
        '''
        用搜索的方法查找邻近点，不方便判断边界点，全部用加权最小二乘法评估
        '''
        self.train_X = X
        self.train_y = y
        
        self.train_number = X.shape[0]
        self.X_dim  = X.shape[1]
        
        self.neighbor = NearestNeighbors(
            n_neighbors = self.k, 
            metric      = self.distance, 
            p           = self.p
        )
        self.neighbor.fit(X, y)
        return None
    # --------------------------------------------
    
    def __general_predict(self, X):
        '''
        用搜索的方法来评估
        '''
        neigh_dist, neigh_inx = self.neighbor.kneighbors(X)
        
        b_0 = np.ones((self.X_dim+1))
        matirx_X = np.ones((self.k, self.X_dim+1))
        
        pred = []
        for ix in range(X.shape[0]):
            x_0 = X[ix]
            x_0_neighbor_indices = neigh_inx[ix]
            x_0_neighbor = self.train_X[x_0_neighbor_indices]
            y_0_neighbor = self.train_y[x_0_neighbor_indices]
            # 距离
            dist = neigh_dist[ix]
            kernel_weight = self.__kernelFunction(dist / np.max(dist))
            
            try:
                matrix_W = np.diag(kernel_weight)
                matirx_X[:, 1:] = x_0_neighbor
                inv = np.linalg.inv(np.dot(np.dot(matirx_X.T, matrix_W), matirx_X))
            except:
                print('对边界点估计时，矩阵 X.T*W*X 不可逆')
                return None
            
            b_0[1:] = x_0
            # 预测
            y_0 = np.dot(np.dot(np.dot(np.dot(b_0.T, inv), matirx_X.T), matrix_W), y_0_neighbor)      
            
            pred.append(y_0)
        
        return np.array(pred)
    # --------------------------------------------
    
    def fit(self, X, y):
        '''
        训练
        '''
        self.__judge_input_dims(X)
        
        if self.istimeseries:
            self.__timeseries_fit(X, y)
        else:  
            self.__general_fit(X, y)
            
        # 完成训练
        self.istrained = True
        return self
    # --------------------------------------------  
    
    def __timeseries_predict(self, x_0):
        '''
        对于单点的 x_0 平滑
        '''
        x_0_neighbor_indices = self.__timeseries_findNeighbor_index(x_0)
        x_0_neighbor = self.train_x[x_0_neighbor_indices]
        y_0_neighbor = self.train_y[x_0_neighbor_indices]
        
        dist = np.abs(x_0 - x_0_neighbor)
        kernel_weight = self.__kernelFunction(dist / np.max(dist))
        
        if self.x_0_position == 1:
            y_0 = np.dot(kernel_weight, y_0_neighbor)/np.sum(kernel_weight)
        else:
            try:
                matrix_W = np.diag(kernel_weight)
                matirx_X = np.ones((self.k, 2))
                matirx_X[:, 1] = x_0_neighbor
                inv = np.linalg.inv(np.dot(np.dot(matirx_X.T, matrix_W), matirx_X))
            except:
                print('对边界点估计时，矩阵 X.T*W*X 不可逆')
                return None
            b_0 = np.array([1, x_0])
            y_0 = np.dot(np.dot(np.dot(np.dot(b_0.T, inv), matirx_X.T), matrix_W), y_0_neighbor)
        return y_0
    # --------------------------------------------     
    
    def predict(self, X):
        '''
        预测
        '''
        self.__judge_input_dims(X)
        
        try:
            assert self.istrained
        except:
            print('模型尚未训练！')
            return None
        
        if self.istimeseries:
            pred = []
            for x_0_index in range(X.size):
                x_0 = X[x_0_index]
                pred.append(self.__timeseries_predict(x_0))
            pred = np.array(pred)
        else:
            pred = self.__general_predict(X)
        return pred
    # --------------------------------------------  
# ================================================================================================