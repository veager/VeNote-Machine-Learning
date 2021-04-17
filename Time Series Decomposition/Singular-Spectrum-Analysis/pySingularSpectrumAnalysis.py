import numpy as np

class SingularSpectrumAnalysis():
    
    def __init__(self, L):
        '''
        series : np.ndarray
        '''
        
        # 窗口宽度
        self.L = L
        
        self.isdecomposed = False
        
        return None
    # -----------------------------------------------------------------------------  
    def decompose_base(self, x):
        '''
        
        '''
        # 序列长度
        T = x.size

        # 向量个数
        N = T - self.L + 1
        
        # 步骤 1 嵌入
        Matrix_F = np.array([x[i:self.L+i] for i in range(0, N)]).T
        
        # 步骤 2 SVD
        Matrix_U, self.sigma, Matrix_VT = np.linalg.svd(Matrix_F)
        # sigma 是奇异值（singular value），等于np.sqrt(lanmda)
        d = np.linalg.matrix_rank(Matrix_F)
        Matrix_V =  Matrix_VT.T
        
        # 步骤 3 重组
        # pass 
        
        # 步骤 4 对角平均
        ts_comps = np.zeros((T, d))
        # 
        Matrix_F_elem = np.array([self.sigma[i]*np.outer(Matrix_U[:,i], Matrix_VT[i,:]) for i in range(d)])
        # 注: np.outer 外积, 两个向量的外积是一个矩阵
        for i in range(d):
            F_rev = Matrix_F_elem[i, ::-1]
            ts_comps[:, i] = [F_rev.diagonal(j).mean() for j in range(-F_rev.shape[0]+1, F_rev.shape[1])]
        return ts_comps
    # -----------------------------------------------------------------------------  

    def decompose(self, x):
        '''
        
        '''
        if not isinstance(x, np.ndarray):
            raise TypeError('输入数据 series 类型必须为 np.ndarray 类型')
            
        if not x.ndim == 1:
            raise ValueError('输入数据 series 的维度必须为 1 维')

        # 原始样本
        self.train_x = x
        # 序列长度
        self.train_len = x.size
        
        if not 2 <= self.L <= int(self.train_len/2):
            raise ValueError("窗口宽度 L 必须在区间 [2, int(T/2)]")
        
        # 分解
        self.ts_comps = self.decompose_base(x)

        # 一些用于分析的变量
        # 特征值 = 奇异值 ** 2
        self.eigenvalue = self.sigma * self.sigma
        # 贡献度
        self.contri     = self.eigenvalue / np.sum(self.eigenvalue)
        # 累计贡献度
        self.cum_contri = np.cumsum(self.contri)
        

        self.isdecomposed = True
        
        return self.ts_comps
    # -----------------------------------------------------------------------------
    
    def trend_periodic_recombine(self, threshold=0.95, periodic_end_ind=None):
        '''
        趋势周期重组
        
        Parameters:
        -----------
        threshold: float, in the bound of [0., 1.]
        阈值 
        
        periodic_end_ind: int

        '''
        # 先分解
        assert self.isdecomposed
        
        if periodic_end_ind is None:
            for i in range(self.cum_contri.shape[0]):
                if self.cum_contri[i] >= threshold:
                    periodic_end_ind = i
                    break

        # 趋势成分 Trend
        self.T_t = self.ts_comps[:, 0]
        # 周期成分 periodic
        self.P_t = np.sum(self.ts_comps[:, 1:periodic_end_ind+1], axis=1)
        # 残差成分 remainder
        self.R_t = np.sum(self.ts_comps[:, periodic_end_ind+1:], axis=1)
        
        # periodic 成分的终止（包括）索引
        self.periodic_end_ind = periodic_end_ind

        return self.T_t, self.P_t, self.R_t

    # -----------------------------------------------------------------------------
    def pseudo_ofs_decompose(self, x):
        '''
        伪包外分解，为了兼容
        '''
        if not isinstance(x, np.ndarray):
            raise TypeError('输入数据 series 类型必须为 np.ndarray 类型')
            
        if not x.ndim == 1:
            raise ValueError('输入数据 series 的维度必须为 1 维')

        x = np.append(self.train_x, x)
        ts_comps = self.decompose_base(x)

        self.ofs_T_t = ts_comps[:, 0]
        self.ofs_P_t = np.sum(ts_comps[:, 1:self.periodic_end_ind+1], axis=1)
        self.ofs_R_t = np.sum(ts_comps[:, self.periodic_end_ind+1:], axis=1)

        self.ofs_T_t = self.ofs_T_t[self.train_len:]
        self.ofs_P_t = self.ofs_P_t[self.train_len:]
        self.ofs_R_t = self.ofs_R_t[self.train_len:]

        return self.ofs_T_t, self.ofs_P_t, self.ofs_R_t
# =================================================================================