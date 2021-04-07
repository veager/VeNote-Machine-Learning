

import math

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F





class PytorchLogitRegression():
    # 参考 pytoch Tutorials
    # https://pytorch.org/tutorials/beginner/nn_tutorial.html
    def __init__(self, batch_size=512, epochs=500, verbose=100, learning_rate=0.001):
        
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.verbose       = verbose
        self.learning_rate = learning_rate
        
        return None
    # -------------------------------------------------------------------------
    def log_softmax(self, data_X):
        # print('x', x.shape)
        # print('x之后', x.exp().sum(-1).log().unsqueeze(-1).shape)
        return data_X - data_X.exp().sum(-1).log().unsqueeze(-1)
        # 这里 log_softmax 设定 log 是以自然对数为底的对数函数, 进行了进一步的推导
    # -------------------------------------------------------------------------
    def model(self, data_X):
        # 返回的是对数 概率,非真实概率
        return self.log_softmax(data_X @ self.weights + self.bias)
        # 对于两个一维向量来说 xb @ weights 等价于 xb.dot(weights)
        # 对于一个二维矩阵（@之前）和一个一维向量（@之后）, @ 可以实现广播运算
    # -------------------------------------------------------------------------
    def loss_func(self, pred_target_ll, target):
        # pred_target_prob 实际值对应的 log-likehood 值
        # target 目标值 0,1,2,...K-1
        return -pred_target_ll[range(target.shape[0]), target].mean()
    # -------------------------------------------------------------------------
    def accuracy(self, pred_prob, train_y):
        preds = torch.argmax(pred_prob, dim=1)
        return (preds == train_y).float().mean()
    # -------------------------------------------------------------------------
    def fit(self, train_X, train_y, valid_X=None, valid_y=None):
        
        train_X, train_y = map(torch.tensor, (train_X, train_y))
        if not (valid_X is None):
            valid_X, valid_y = map(torch.tensor, (valid_X, valid_y))
        
        n_sample_train_X = train_X.shape[0]
        
        self.input_dim  = train_X.shape[1]
        self.output_dim = 10
        
        # 权重参数初始化
        self.weights = torch.randn(self.input_dim, self.output_dim) / math.sqrt(self.input_dim)
        self.weights.requires_grad_()
        self.bias = torch.zeros(10, requires_grad=True)
        
        for epoch in range(self.epochs):
            
            for i in range((n_sample_train_X - 1) // self.batch_size + 1):
                # set_trace()
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size
                
                train_X_bs = train_X[start_i :end_i]
                train_y_bs = train_y[start_i :end_i]
                
                # 计算属于每一分类的 对数概率
                pred = self.model(train_X_bs)
                loss = self.loss_func(pred, train_y_bs)
                
                loss.backward()
                
                with torch.no_grad():
                    self.weights -= self.weights.grad * self.learning_rate
                    self.bias -= self.bias.grad * self.learning_rate
                    
                    self.weights.grad.zero_()
                    self.bias.grad.zero_()
            
            if epoch % self.verbose == 0:
                pred = self.model(train_X)
                loss = self.loss_func(pred, train_y).item()
                acc  = self.accuracy(pred, train_y).item()
                print('t:{0}, loss:{1}, acc:{2}'.format(epoch, loss, acc))
        
        return None
    # -------------------------------------------------------------------------
    def pred(self, data_X, prob=False):
        
        # pred = self.model(data_X).exp() / self.model(data_X).exp().sum(-1).log().unsqueeze(-1)
        
        # if not prob:
        #     pred = torch.argmax(pred_prob, dim=1)
            
        return pred
    # -------------------------------------------------------------------------
# =============================================================================









class Mnist_Logistic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 自定义建立线全连接层
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) / math.sqrt(input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        # 使用pytorch预制函数建立全连接层
        # self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, data_X):
        out = data_X @ self.weights + self.bias
        
        # out = self.linear(data_X)
        return out
# =============================================================================


class PytorchModelLogitRegression():
    # 参考 pytoch Tutorials
    # https://pytorch.org/tutorials/beginner/nn_tutorial.html
    def __init__(self, batch_size=512, epochs=500, verbose=100, learning_rate=0.001):
        
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.verbose       = verbose
        self.learning_rate = learning_rate
        
        return None
    # -------------------------------------------------------------------------
    def accuracy(self, pred_prob, train_y):
        preds = torch.argmax(pred_prob, dim=1)
        return (preds == train_y).float().mean()
    # -------------------------------------------------------------------------
    def fit(self, train_X, train_y, valid_X=None, valid_y=None):
        
        train_X, train_y = map(torch.tensor, (train_X, train_y))
        if not (valid_X is None):
            valid_X, valid_y = map(torch.tensor, (valid_X, valid_y))
        
        n_sample_train_X = train_X.shape[0]
        
        self.input_dim  = train_X.shape[1]
        self.output_dim = 10
        
        # 定义模型
        self.model = Mnist_Logistic(self.input_dim, self.output_dim)
        # 交叉熵损失函数
        self.loss_func = F.cross_entropy
        
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            for i in range((n_sample_train_X - 1) // self.batch_size + 1):
                # set_trace()
                start_i = i * self.batch_size
                end_i = start_i + self.batch_size
                
                train_X_bs = train_X[start_i :end_i]
                train_y_bs = train_y[start_i :end_i]
                
                # 计算属于每一分类的 对数概率
                pred = self.model(train_X_bs)
                loss = self.loss_func(pred, train_y_bs)
                
                loss.backward()
                
                opt.step()
                opt.zero_grad()
                # 上述两行等价于
                # with torch.no_grad():
                #     for p in self.model.parameters():
                #         p -= p.grad * self.learning_rate
                #     self.model.zero_grad()
                
            if epoch % self.verbose == 0:
                pred = self.model(train_X)
                loss = self.loss_func(pred, train_y).item()
                acc  = self.accuracy(pred, train_y).item()
                print('t:{0}, loss:{1}, acc:{2}'.format(epoch, loss, acc))
        
        return None
    # -------------------------------------------------------------------------
    def pred(self, data_X, prob=False):
        
        # pred = self.model(data_X).exp() / self.model(data_X).exp().sum(-1).log().unsqueeze(-1)
        
        # if not prob:
        #     pred = torch.argmax(pred_prob, dim=1)
            
        return pred
    # -------------------------------------------------------------------------
# =============================================================================







# 加载 MINIT数据集
from pathlib import Path

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"

PATH.mkdir(parents=True, exist_ok=True)

import pickle
import gzip
# 加载数据
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")



