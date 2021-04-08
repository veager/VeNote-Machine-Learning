import numpy as np
from sklearn.neighbors import NearestNeighbors

class PeriodicTrendDecomposition():
    def __init__(self, period_length, k1, k2, k3, n_inner=5, n_outer=5, robust=True, season_norm=True):
        '''
        '''
        self.period_length = period_length
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        
        self.n_inner = n_inner
        self.n_outer = n_outer
        
        self.robust  = robust
        if not robust: # robust == False
            # 当不使用鲁棒权重时，外循环次数被设为 1
            if self.n_outer != 1:
                print('robust:{0}, 将不计算鲁棒权重，外循环次数n_outer将被设为1'.format(robust))
                self.n_outer = 1
        self.season_norm = season_norm
        
        return None
    # --------------------------------------------------------
    
    def transToSubseries(self, x):
        '''
        转换为周期子序列
        '''
        subseries = {}
        if isinstance(x, list):
            x_length = len(x)
        elif isinstance(x, np.ndarray):
            x_length = x.size
        else:
            print(type(x))
            return None
        
        for i in range(self.period_length):
            subseries[i] = x[i::self.period_length]
        return subseries
    # --------------------------------------------------------
    
    def subseriesSmooth(self, subseries, k):
        '''
        subseries: dict, 由self.subseriesSmooth()函数得到的周期子序列
        '''

        pred_subseris = []
        for i, y in subseries.items():
            pred_subseris_i, self.subseris_neigh_dist[i], self.subseris_neigh_ind[i] = self.LOESSSmooth(
                y               = np.array(y), 
                k               = k, 
                neigh_dist      = self.subseris_neigh_dist[i],
                neigh_ind       = self.subseris_neigh_ind[i],
                y_robust_weight = self.subseries_robust_weight[i],
                expand          = True
            )
            pred_subseris.append(pred_subseris_i)
        
        C_t = []
        # print(self.n_period, len(pred_subseris[0]))
        for i in range(self.n_period+2):
            for pred_subseris_i in pred_subseris:
                C_t.append(pred_subseris_i[i])
        return C_t
    # --------------------------------------------------------
    
    def moveAverage(self, x, width):
        return [np.mean(x[i:i+width]) for i in range(len(x)-width)]
    # --------------------------------------------------------
    
    def seasonNormalization(self, x):
        '''
        将季节成分转换为严格周期函数
        '''
        P_t = [np.mean(x[i::self.period_length]) for i in range(self.period_length)]
        S_t = P_t * self.n_period
        return S_t, P_t
    # --------------------------------------------------------
    
    def kernelFuction(self, x, name='bisquare'):
        '''
        x 为 list
        '''
        if isinstance(x, np.ndarray):
            y = np.clip(x, -1., 1.)
            y = (1. - y ** 2) ** 2
        elif isinstance(x, list):
            y = map(x, lambda i: min(max(-1., i), 1.))
            y = map(y, lambda i: (1. - y*y)**2)
            y = list(y)
        elif isinstance(x, float) or isinstance(x, int):
            y = min(max(-1., x), 1.)
            # print('x', x, y)
            y = (1. - y*y)**2
        else:
            return None
        return y
    # --------------------------------------------------------
    
    def x_expeand(self, x, degree=2):
        '''
        增加维度
        '''
        if isinstance(x, np.ndarray):
            # sklearn.preprocessing.PolynomialFeatures 对象
            if degree == 0:
                out = np.ones(x.shape[0], dtype=float)

            elif degree == 1:
                out = np.ones((x.shape[0], 2), dtype=float)
                out[:, 1] = x

            elif degree >= 2:
                out = np.ones((x.shape[0], degree+1), dtype=float)
                for i in range(1, 1+degree):
                    out[:, i] = x ** i
        elif isinstance(x, float) or isinstance(x, int):
            out = np.array([x**i for i in range(degree+1)])
        else:
            return None
        return out
    
    # --------------------------------------------------------
    def LOESSSmooth(self, y, k, neigh_dist, neigh_ind, y_robust_weight, expand=False):
        '''
        LOESS 平滑
        默认 x = [0, 1, ..., len(y)]
        对 x 所有的点进行 LOESS 估计
        
        y, y_robust_weight: list 类型
        neigh_ind, y_robust_weight: np.ndarray 类型
        '''
        # 统一的参数
        robust = self.robust
        
        try:
            assert len(y) == len(y_robust_weight)
        except:
            print('LOESSSmooth 函数的输入变量 y 与 y_robust_weight 的长度不同')
        
        # assert y.ndim == 1
        # 训练合集
        x_train = list(range(len(y)))
        # print(np.array(x).reshape(-1, 1))
        # print(x, len(x))
        
        if expand:
            # 预测的 x 合集
            x_pred = [-1] + x_train + [len(x_train)]
        else:
            x_pred = x_train
        
        # 列表转换为 np.ndarray形式
        x_train = np.array(x_train)
        x_pred = np.array(x_pred)
        y = np.array(y)
        y_robust_weight = np.array(y_robust_weight)
        
        if neigh_dist is None: # neigh_dist = None
            neigh_model = NearestNeighbors(n_neighbors=k, metric='manhattan', n_jobs=-1)
            # 训练
            neigh_model.fit(x_train.reshape(-1, 1), y)
            # 查找
            neigh_dist, neigh_ind = neigh_model.kneighbors(x_pred.reshape(-1, 1))
            # print(neigh_dist.shape)
        
        try:
            assert neigh_dist.shape[0] == x_pred.shape[0]
        except:
            print(neigh_dist.shape[0],  x_pred.shape[0])
        
        # 相当于在 LOESS 平滑中 degree = 1
        x_train_expand = np.ones((x_train.shape[0], 2), dtype=float)
        x_train_expand[:, 1] = x_train
        
        x_pred_expand = np.ones((x_pred.shape[0], 2), dtype=float)
        x_pred_expand[:, 1] = x_pred
        
        y_pred = []
        for i in range(len(x_pred)):
            # 目标点
            x0_pred = x_pred_expand[i]
            # 邻近点索引
            x0_neigh_ind = neigh_ind[i]
            
            matrix_X = x_train_expand[x0_neigh_ind]
            # print(y)
            matrix_Y = y[x0_neigh_ind]
            # 距离
            x0_neigh_dist = neigh_dist[i]
            # 核权重
            kernel_weight = self.kernelFuction(x0_neigh_dist / np.max(x0_neigh_dist), name='bisquare') 
            # 总权重
            # print(type(y_robust_weight))
            # print(x0_neigh_ind)
            if robust:
                weight = kernel_weight*y_robust_weight[x0_neigh_ind]
            else:
                weight = kernel_weight
            # 预
            matrix_W = np.diag(weight)
            try:
                inv = np.linalg.inv(np.dot(np.dot(matrix_X.T, matrix_W), matrix_X))
            except:
                print('对边界点{0}估计时，矩阵 X.T*W*X 不可逆'.format(x0_pred))
                print(matrix_X)
                print(matrix_W)
                print(weight, kernel_weight, y_robust_weight[x0_neigh_ind])
                return None

            # 预测
            y0 = np.dot(np.dot(np.dot(np.dot(x0_pred.T, inv), matrix_X.T), matrix_W), matrix_Y)
            
            # 
            y_pred.append(y0)
        
        return y_pred, neigh_dist, neigh_ind
    # --------------------------------------------------------
    
    def LOESSforSingle(self, x0, x_train, y_train, y_robust_weight, degree=1):
        '''
        x : float or int. 目标点
        x_train : list. 邻近点x 
        y_train : list. 邻近点y
        neigh_dist : list. 邻近距离
        degree : int. 取 1 或 2 比较合适
        '''
        assert len(y_train) == self.k3
        
        # 相当于在 LOESS 平滑中 degree = 1
#         x0_pred = self.x_expeand(np.array(x0), degree=degree)
        x0_pred = self.x_expeand(x0, degree=degree)
        # matrix_X = np.ones((len(x_train), 2), dtype=float)
        # matrix_X[:, 1] = x_train
        matrix_X = self.x_expeand(np.array(x_train), degree=degree)
        matrix_Y = np.array(y_train)
        y_robust_weight = np.array(y_robust_weight)
        # print(matrix_X, x0_pred)
        # 邻近距离根据时间序列推测
        x0_neigh_dist = self.k3 - 1. - np.arange(len(y_train))

        # 核权重
        kernel_weight = self.kernelFuction(x0_neigh_dist / np.max(x0_neigh_dist), name='bisquare') 
        # 全部权重
        if self.robust:
            weight = kernel_weight * y_robust_weight
        else:
            weight = kernel_weight
        
        matrix_W = np.diag(weight)
        try:
            inv = np.linalg.inv(np.dot(np.dot(matrix_X.T, matrix_W), matrix_X))
        except:
            print('LOESSforSingle: 对边界点{0}估计时，矩阵 X.T*W*X 不可逆'.format(x0_pred))
            print(matrix_X)
            print('matrix_W:',      matrix_W)
            print('x0_neigh_dist:', x0_neigh_dist)
            print('weight:',        weight)
            print('kernel_weight:', kernel_weight)
            print('y_robust_weight:', y_robust_weight)
            return None

        # 预测
        y0 = np.dot(np.dot(np.dot(np.dot(x0_pred.T, inv), matrix_X.T), matrix_W), matrix_Y)
        
        return y0
    # --------------------------------------------------------
    
    def __inner_loop(self, x):
        '''
        
        '''
        # Step 1:
        self.T_t_detrend = [a - self.T_t[i] for i, a in enumerate(x)]
        
        # Step 2:
        # 转换为周期子序列
        self.subseries = self.transToSubseries(self.T_t_detrend)
        self.subseries_robust_weight = self.transToSubseries(self.robust_weight)
        # print(type(self.subseries_robust_weight[1]))
        # print(self.subseries)
        # 周期子序列平滑
        self.C_t = self.subseriesSmooth(self.subseries, k=self.k1)
        
        # Step 3:
        self.E_t = self.moveAverage(self.C_t, self.period_length)
        self.E_t = self.moveAverage(self.E_t, self.period_length)
        # # LOEES 平滑
        self.E_t, self.E_t_neigh_dist, self.E_t_neigh_ind = self.LOESSSmooth(
            y               = self.E_t, 
            k               = self.k2, 
            neigh_dist      = self.E_t_neigh_dist,
            neigh_ind       = self.E_t_neigh_ind,
            y_robust_weight = self.robust_weight
        )
        
        # Step 4: 计算季节成分
        # print(len(self.C_t))
        self.S_t =  [self.C_t[i+self.period_length] - self.E_t[i] for i in range(self.series_length)]
        if self.season_norm:
            self.S_t, self.P_t = self.seasonNormalization(self.S_t)
        
        # Step 5: 去季节成分
        # print(len(x), len(self.S_t))
        self.S_t_deseason = [a - self.S_t[i] for i, a in enumerate(x)]
        
        # Step 6:  计算趋势成分
        self.T_t, self.S_t_deseason_neigh_dist, self.S_t_deseason_neigh_ind = self.LOESSSmooth(
            y               = self.S_t_deseason, 
            k               = self.k3, 
            neigh_dist      = self.S_t_deseason_neigh_dist,
            neigh_ind       = self.S_t_deseason_neigh_ind,
            y_robust_weight = self.robust_weight
        )
        return None
    # --------------------------------------------------------
    
    def decompose(self, x):
        '''
        Parameters:
        -----------
        x : list
        
        Return:
        -------
        T_t, S_t, R_t : list
        
        '''
        # 序列长度
        self.series_length = len(x)
        # 周期数量 不足一个周期的 则 舍去
        self.n_period = self.series_length // self.period_length
        
        # 初始 样本鲁棒权重
        self.robust_weight = [1.] *  self.series_length
        
        # 初始 STEP 2 中 Subseries 邻近索引、邻近距离
        self.subseris_neigh_dist = [None] * self.period_length
        self.subseris_neigh_ind = [None] * self.period_length
        
        # 初始 STEP 3 中 平滑周期子序列 邻近索引、邻近距离
        self.E_t_neigh_dist, self.E_t_neigh_ind = None, None
        
        # 初始 STEP 6 中 去季节序列 邻近索引、邻近距离
        self.S_t_deseason_neigh_dist, self.S_t_deseason_neigh_ind = None, None
        
        # 初化 趋势序列
        self.T_t = [0.] * self.series_length
        
        for i_outer in range(self.n_outer):
            for i_inner in range(self.n_inner):
                # print(i_inner)
                self.__inner_loop(x)
            self.R_t = [a - self.T_t[i] - self.S_t[i] for i, a in enumerate(x)]
            
            if self.robust:
                self.robust_weight_median = np.median(np.abs(self.R_t))
                self.robust_weight = self.kernelFuction(np.abs(self.R_t) / (6*self.robust_weight_median))
                self.robust_weight = list(self.robust_weight)
        return self.T_t, self.S_t, self.R_t
    # --------------------------------------------------------
    
    def outofsampelDecompose_single(self, x):
        '''
        '''
        # Step 1: 计算季节成分
        # print(self.x_pred_ix % self.period_length)
        ofs_S_i = self.P_t[self.x_pred_ix % self.period_length]
        
        # Step 2: 计算去季节成分
        ofs_S_i_deseason = x - ofs_S_i
        # 更新全部的去季节成分
        self.S_t_deseason[-1] = ofs_S_i_deseason
        
        # Step 3: 计算趋势成分
        # 使用LOESS估计时, 将x整体平移值从0开始
        x0      = self.series_length + self.x_pred_ix
        x_train = np.arange(x0-self.k3, x0) + 1
        # 
        y_train         = self.S_t_deseason[-self.k3:]
        y_robust_weight = self.robust_weight[-self.k3:]
        # print(len(x_train), x_train, x0, len(x_train), len(y_train), len(y_robust_weight))
        ofs_T_i = self.LOESSforSingle(x0, x_train, y_train, y_robust_weight)
        
        # Step 4: 计算残差成分
        ofs_R_i = x - ofs_T_i - ofs_S_i
        
        return ofs_T_i, ofs_S_i, ofs_R_i, ofs_S_i_deseason
    # --------------------------------------------------------
    
    def outofsampelDecompose(self, x):
        '''
        Parameters:
        -----------
        x : list
        
        Return:
        -------
        ofs_T_t, ofs_S_t, ofs_R_t : list
        
        '''
        # 只对 规整化的 周期成分可用
        assert self.season_norm == True
        
        self.ofs_T_t, self.ofs_S_t, self.ofs_R_t = [], [], []
        # 初始时的鲁棒权重
        self.ofs_robust_weight = []
        
        self.S_t_deseason.append(0.)
        self.robust_weight.append(1.)
        
        # 包外样本的索引 用于判断时间位置，计算周期成分
        self.x_pred_ix = 0
        
        for ele_i, ele_v in enumerate(x):
            # print('ele', ele_i, self.x_pred_ix % self.period_length)
            for i_outer in range(self.n_outer):
                # ofs_n_inner = 1 
                # 包外分解，内循环不存在更新
                ofs_n_inner = 1
                for i_inner in range(ofs_n_inner):
                    
                    ofs_T_i, ofs_S_i, ofs_R_i, ofs_S_i_deseason = self.outofsampelDecompose_single(ele_v)
                
                # 内循环结束
                if self.robust:
                    # 计算新增样本鲁棒权重
                    # kernel_x = abs(ofs_R_i) / (6*self.robust_weight_median)
                    # print('kernel_x', kernel_x, self.kernelFuction(kernel_x))
                    # print(self.kernelFuction(abs(ofs_R_i) / (6*self.robust_weight_median)))
                    self.robust_weight[self.series_length + self.x_pred_ix] = self.kernelFuction(abs(ofs_R_i) / (6*self.robust_weight_median)) 
            # 单个 样本估计 结束
            self.x_pred_ix += 1
            # 更新数据
            self.ofs_T_t.append(ofs_T_i), self.ofs_S_t.append(ofs_S_i), self.ofs_R_t.append(ofs_R_i)
            
            self.robust_weight.append(1.)
            self.S_t_deseason.append(ofs_S_i_deseason)
            # print(self.robust_weight[-2:])
        self.robust_weight.pop(-1)
        
        return self.ofs_T_t, self.ofs_S_t, self.ofs_R_t
    # --------------------------------------------------------