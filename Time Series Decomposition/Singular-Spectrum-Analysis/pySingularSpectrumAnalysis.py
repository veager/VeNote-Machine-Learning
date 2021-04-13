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
    
    def decomposition(self, series):
        '''
        
        '''
        if not isinstance(series, np.ndarray):
            raise TypeError('输入数据 series 类型必须为 np.ndarray 类型')
            
        if not series.ndim == 1:
            raise ValueError('输入数据 series 的维度必须为 1 维')
        
        # 序列长度
        self.T = len(series) 
        
        if not 2 <= self.L <= int(self.T/2):
            raise ValueError("窗口宽度 L 必须在区间 [2, int(T/2)]")
        
        # 向量个数
        self.N = self.T - self.L + 1
        
        # 步骤 1 嵌入
        self.F = np.array([series[i:self.L+i] for i in range(0, self.N)]).T
        
        # 步骤 2 SVD
        self.U, self.sigma, VT = np.linalg.svd(self.F)
        # sigma 是奇异值（singular value），等于np.sqrt(lanmda)
        self.d = np.linalg.matrix_rank(self.F)
        self.V = VT.T
        
        # 步骤 3 重组
        # pass 
        
        # 步骤 4 对角平均
        self.ts_comps = np.zeros((self.T, self.d))
        # 
        
        # 一些用于分析的变量
        # 特征值 = 奇异值 ** 2
        self.eigenvalue = self.sigma * self.sigma
        # 贡献度
        self.contri     = self.eigenvalue / np.sum(self.eigenvalue)
        # 累计贡献度
        self.cum_contri = np.cumsum(self.contri)
        
        
        self.F_elem = np.array([self.sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d)])
        # 注: np.outer 外积, 两个向量的外积是一个矩阵
        for i in range(self.d):
            F_rev = self.F_elem[i, ::-1]
            self.ts_comps[:, i] = [F_rev.diagonal(j).mean() for j in range(-F_rev.shape[0]+1, F_rev.shape[1])]
        
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
        periodic 成分的起始（包括）和终止（不包括）索引
        
        
        '''
        assert self.isdecomposed
        
        if periodic_end_ind is None:
            for i in range(self.cum_contri.shape[0]):
                if self.cum_contri[i] >= threshold:
                    periodic_end_ind = i
                    break
        # 趋势成分
        self.ts_recombie = self.ts_comps[:, :3].copy()
        # 周期成分
        self.ts_recombie[:, 1] = np.sum(self.ts_comps[:, 1:periodic_end_ind+1], axis=1)
        # 残差成分
        self.ts_recombie[:, 2] = np.sum(self.ts_comps[:, periodic_end_ind+1:], axis=1)
        
        return self.ts_recombie, periodic_end_ind
# =================================================================================