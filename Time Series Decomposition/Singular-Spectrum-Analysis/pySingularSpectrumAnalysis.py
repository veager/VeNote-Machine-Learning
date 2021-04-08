import numpy as np

class SingularSpectrumAnalysis():
    
    def __init__(self, L):
        '''
        series : np.ndarray
        '''
        
        # 窗口宽度
        self.L = L
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
        print(self.F.shape)
        # 步骤 2 SVD
        self.U, self.sigma, VT = np.linalg.svd(self.F)
        self.d = np.linalg.matrix_rank(self.F)
        self.V = VT.T
        
        # 步骤 3 重组
        # pass 
        
        # 步骤 4 对角平均
        self.ts_comps = np.zeros((self.T, self.d))
        # 
        self.F_elem = np.array([self.sigma[i]*np.outer(self.U[:,i], VT[i,:]) for i in range(self.d)])
        # 注: np.outer 外积, 两个向量的外积是一个矩阵
        for i in range(self.d):
            F_rev = self.F_elem[i, ::-1]
            self.ts_comps[:, i] = [F_rev.diagonal(j).mean() for j in range(-F_rev.shape[0]+1, F_rev.shape[1])]
        
        return self.ts_comps
    # -----------------------------------------------------------------------------    
# =================================================================================