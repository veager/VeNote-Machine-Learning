import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class ExtremeLearninMachineBase(BaseEstimator):
    def __init__(self, hidden_unit, regularization_factor, activation_type, random_seed):
        '''
        Parameters:
        -----------
        
        '''
        self.hidden_unit  = hidden_unit
        self.regularization_factor = regularization_factor
        self.activation_type = activation_type
        self.random_seed = random_seed
        
        self.istrained = False
    # -------------------------------------------------------------------------
    
    def __activations(self, x):
        '''
        激活函数
        '''
        if self.activation_type == 'sigmoid':
            out = 1. / (1. + np.exp(-x))
        
        elif self.activation_type == 'tanh':
            out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        
        elif self.activation_type == 'relu':
            out = np.clip(x, 0, np.inf)
        
        elif self.activation_type == 'linear':
            out = x
        
        return out
    # -------------------------------------------------------------------------
    
    def __initialization(self):
        # 经试验验证，模型性能与随机数种子，
        # 初始化权重分布（[0,1]或[-1,1]）有很大的关系
        rand = np.random.RandomState(self.random_seed)
        self.w_in = rand.uniform(low=0., high=1., size = (self.input_unit, self.hidden_unit))
        self.b_in = rand.uniform(low=0., high=1., size = self.hidden_unit)
        # self.w_in = rand.normal(loc=0.0, scale=0.5, size=(self.input_unit, self.hidden_unit))
        # self.b_in = rand.normal(loc=0.0, scale=0.5, size=self.hidden_unit)
        return None
    
    # -------------------------------------------------------------------------
    def __hidden_state(self, X):
        return self.__activations(np.dot(X, self.w_in) + self.b_in)
    
    # -------------------------------------------------------------------------
    def train(self, X, y):
        '''      
        
        '''
        # 输入维度 input_unit
        self.input_unit = X.shape[1]
        
        # 输入到隐含层连接初始化
        self.__initialization()
        
        # 计算隐含状态
        H = self.__hidden_state(X)
        
        try: # 求矩阵逆的时候，可能会存在矩阵不可逆的情况
            inv = np.linalg.inv(1 / self.regularization_factor + np.dot(H.T, H))
        except:
            print('矩阵 I/C + H.T*H 不可逆')
            return None
        
        self.beta = np.dot(np.dot(inv, H.T), y)        
        
        # 训练完成
        self.istrained = True
        
        return None
    # -------------------------------------------------------------------------
    # 预测
    def output(self, X):
        '''
        
        '''
        try:
            assert self.istrained
        except:
            print('模型还未训练！')
        
        H = self.__hidden_state(X)
        out = np.dot(H, self.beta)
        return out
    # -------------------------------------------------------------------------
# =============================================================================


#
#
#   用于回归的 极限学习机
# 
#
class ExtremeLearninMachineRegression(RegressorMixin, ExtremeLearninMachineBase):
    
    def __init__(self, hidden_unit=10, regularization_factor=0.5, activation_type='sigmoid', random_seed=42):
        ExtremeLearninMachineBase.__init__(self, hidden_unit, regularization_factor, activation_type, random_seed)
        return None
    # -------------------------------------------------------------------------
    
    def fit(self, X, y):
        '''
        训练
        '''
        # 训练
        self.train(X, y)
        return self
    # -------------------------------------------------------------------------
    
    def predict(self, X):
        '''
        预测
        '''
        return self.output(X)
    # -------------------------------------------------------------------------
# =============================================================================
    
    


#
#
#   用于分类的 极限学习机
# 
#   
class ExtremeLearninMachineClassifier(ClassifierMixin, ExtremeLearninMachineBase):
    
    def __init__(self, hidden_unit=10, regularization_factor=0.5, activation_type='sigmoid', random_seed=42):
        ExtremeLearninMachineBase.__init__(self, hidden_unit, regularization_factor, activation_type, random_seed)
        return None
    # ------------------------------------------------------------------------
    
    def __onehotEncode(self, y):
        '''
        对 y 进行 one-hot 编码
        '''
        # 标签
        self.y_labels = list(np.unique(y)) # 会从小到大排列
        
        enconde = np.zeros((y.shape[0], len(self.y_labels)))
        
        for ix, l in enumerate(self.y_labels):
            enconde[y==l, ix] = 1
        return enconde
    # -------------------------------------------------------------------------
    
    def __softmax(self, x):
        '''
        softmax 输出
        '''
        out = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
        return out
    # -------------------------------------------------------------------------
    
    def fit(self, X, y):
        '''
        训练
        '''
        # y 编码
        y_onehot = self.__onehotEncode(y)
        # 训练
        self.train(X, y_onehot)
        return self
    # -------------------------------------------------------------------------
    
    def predict(self, X):
        '''
        预测
        '''
        pred_proba = self.predict_proba(X)
        pred = np.argmax(pred_proba, axis=1)
        return pred
    # -------------------------------------------------------------------------
    
    def predict_proba(self, X):
        '''
        预测概率
        '''
        out = self.output(X)
        pred_proba = self.__softmax(out)
        return pred_proba
    # -------------------------------------------------------------------------
# =============================================================================