
import time
import numpy as np
import matplotlib.pyplot as plt



class FuzzyCMeanCluster():
    # Fuzzy C Mean 聚类
    def __init__(self, n_cluster, n_iter, m=2):
        
        self.n_cluster = n_cluster # 聚类个数
        self.n_iter = n_iter       # 迭代次数
        self.m = m
    # -------------------------------------------------------------------------
    def ComputeDistMatrix(self, X, Y):
        
        n_sample_X = X.shape[0]
        n_sample_Y = Y.shape[0]
        
        dist_matrix = np.zeros((n_sample_X, n_sample_Y))
        # 计算样本到聚类中心的距离矩阵
        for i in range(n_sample_X):
            for j in range(n_sample_Y):
                # 两个向量的欧式距离
                vector = X[i, :]-Y[j, :]
                # 欧拉距离
                # dist_matrix[i, j] = np.sqrt(np.dot(vector, vector))
                # 与K mean 保持一致
                dist_matrix[i, j] = np.dot(vector, vector)
        return dist_matrix
    # -------------------------------------------------------------------------
    def UpdataCenters(self):
        
        self.mu_m = np.power(self.mu, self.m)
        # 计算分母
        self.mu_m_sum_sample = np.sum(self.mu_m, axis=0).reshape((1, self.n_cluster))
        # print(self.mu_m.shape, self.mu_m_sum_sample.shape)
        # 分数项
        self.mu_m_ = self.mu_m / self.mu_m_sum_sample
        
        for i in range(self.n_cluster):
            self.centers[i, :] = np.sum(self.mu_m_[:, i].reshape((self.n_samples_train, 1)) * self.train, axis=0)
            # print(self.mu_m_[:, i].reshape((self.n_samples_train, 1)).shape)
            # print(self.centers[i, :].shape)
        return None
    # -------------------------------------------------------------------------
    def UpdataMu(self):
        
        self.dist_matrix = self.ComputeDistMatrix(self.train, self.centers)
        
        self.mu = (self.dist_matrix ** (-1/(self.m-1))) / np.sum(self.dist_matrix ** (-1/(self.m-1)), axis=1).reshape((self.n_samples_train, 1))
        # print(np.sum(self.dist_matrix ** (-1/(self.m-1)), axis=1).shape)
        # print(np.sum(self.mu, axis=1))
        return None
    # -------------------------------------------------------------------------
    def Fit(self, train):
        '''
        

        Parameters
        ----------
        train : 待聚类样本, (n_samples, n_dims)
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        assert train.ndim in [2,3]
        # 训练样本个数
        self.n_samples_train = train.shape[0]
        self.n_dims          = train.shape[1]
        
        # 初始化模糊度矩阵
        self.mu = np.random.random((self.n_samples_train, self.n_cluster))
        self.mu = self.mu / np.sum(self.mu, axis=0)
        
        # 初始化聚类中心
        self.centers = np.zeros((self.n_cluster, self.n_dims))
        
        self.train = train
        
        # 初始化中心点
        self.all_centers = np.zeros((self.n_iter, self.n_cluster, self.n_dims))
        # 聚类过程
        for i in range(self.n_iter):
            self.UpdataCenters()
            self.UpdataMu()
            
            self.all_centers[i, :, :] = self.centers
        
        # 计算样本属于哪一个中心
        self.train_belong_index = np.argmax(self.mu, axis=1)
        
        return None
    # -------------------------------------------------------------------------
    def Predict(self, test):
        
        # 测试样本格式
        n_samples_test = test.shape[0]
        # 距离矩阵
        dist_matrix = self.ComputeDistMatrix(test, self.centers)
        # 计算隶属度
        test_mu = (dist_matrix ** (-1/(self.m-1))) / np.sum(self.dist_matrix ** (-1/(self.m-1)), axis=1).reshape((n_samples_test, 1))
        
        test_belong_index = np.argmax(test_mu, axis=1)
        
        return test_belong_index
    # -------------------------------------------------------------------------





