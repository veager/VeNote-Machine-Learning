
import numpy as np
import torch

class NumpyNeuralNetwork():
    def __init__(self, hidden_unit, learning_rate=0.0001, epochs=500, verbose=100):
        
        self.hidden_unit   = hidden_unit
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.verbose       = verbose
        
        self.activation_name = 'relu'
        
        self.loss_history_train = []
        self.loss_history_valid = []
        
        return None
    # -------------------------------------------------------------------------
    def activation(self, x):
        
        if self.activation_name == 'relu':
            # relu 激活函数
            out = np.maximum(x, 0)
        elif self.activation_name == 'sigmoid':
            # sigmoid 激活函数
            out = 1 / (1 + np.exp(-x))
        elif self.activation_name == 'tanh':
            # tanh 激活函数
            out = np.tanh(x)
        
        return out
    # -------------------------------------------------------------------------
    def activation_derivative(self, x):
        
        if self.activation_name == 'relu':
            # relu 激活函数 导数
            out = x.copy()
            out[out<0] = 0
        elif self.activation_name == 'sigmoid':
            # sigmoid 激活函数
            out = np.exp(-x) / ((1 + np.exp(-x)) * (1 + np.exp(-x)))
        elif self.activation_name == 'tanh':
            # tanh 激活函数
            out = np.sinh(x) / np.cosh(x)
            
        return out
    # -------------------------------------------------------------------------
    # def initailize(self):
        
    #     return None
    # -------------------------------------------------------------------------
    def fit(self, train_X, train_y, valid_X=None, valid_y=None):
        
        assert train_y.ndim == 2
        
        self.n_sample_train = train_X.shape[0]
        
        self.input_unit  = train_X.shape[1]
        self.output_unit = train_y.shape[1]
        # Randomly initialize weights
        # [0, 1]均匀分布初始化
        self.w1 = np.random.rand(self.input_unit,  self.hidden_unit)
        self.w2 = np.random.rand(self.hidden_unit, self.output_unit)
        
        self.b1 = np.random.rand(self.hidden_unit)
        self.b2 = np.random.rand(self.output_unit)
        
            
        # 训练过程
        for i in range(self.epochs):
            
            pred_y = self.pred(train_X)
            
            loss = np.square(pred_y - train_y).sum()
            
            # 训练集 损失函数日志
            self.loss_history_train.append(loss)
            
            # 训练过程输出日志
            if self.verbose == 0:
                pass
            elif i % self.verbose == 0: 
                print('t:{0:>4d}, loss(mse):{1:>12.6f}'.format(i, loss))
            
            self.grad_y = 2.0 * (pred_y - train_y)
            # print('grad_y:', self.grad_y.shape) 
            
            self.delta_2 = self.activation_derivative(self.data_h2_linear)
            self.delta_2 = self.delta_2 * self.grad_y
            # print('delta_2:', self.delta_2.shape)
            
            self.w2_grad = np.dot(self.data_h1_act.T, self.delta_2) / self.n_sample_train
            # 除以样本数self.n_sample_train是求 所有样本平均误差
            self.b2_grad = np.mean(self.delta_2)
            # print('w2_grad:', self.w2_grad.shape)
            
            self.delta_1 = self.activation_derivative(self.data_h1_linear)
            # print('delta_1:', self.delta_1.shape)
            self.delta_1 = self.delta_1 * np.dot(self.delta_2, self.w2.T)
            
            self.w1_grad = np.dot(train_X.T, self.delta_1) / self.n_sample_train
            self.b1_grad = np.mean(self.delta_1)
            
            # 更新参数
            self.w1 = self.w1 - self.learning_rate * self.w1_grad
            self.b1 = self.b1 - self.learning_rate * self.b1_grad
            self.w2 = self.w2 - self.learning_rate * self.w2_grad
            self.b2 = self.b2 - self.learning_rate * self.b2_grad
            
            # 验证集 损失函数日志, 注意: 调用self.pred() 会更新相关参数
            if not(valid_X is None):
                self.loss_history_valid.append(np.square(self.pred(valid_X) - valid_y).sum())
            
        return None
    # ------------------------------------------------------------------------- 
    def pred(self, data_X):
        
        # 隐含层 输出
        self.data_h1_linear = np.dot(data_X, self.w1) + self.b1.reshape(1, -1)
        # print(self.data_h1_linear.shape, self.b1.reshape(1, -1).shape)
        self.data_h1_act    = self.activation(self.data_h1_linear)
        # print('h1_activation', self.data_h1_act.shape)
        
        # 输出层 输出
        self.data_h2_linear = np.dot(self.data_h1_act, self.w2) + self.b2.reshape(1, -1)
        # print(self.data_h2_linear.shape, self.b2.reshape(1, -1).shape)
        self.data_h2_act    = self.activation(self.data_h2_linear)
        # print('h2_activation', self.data_h2_act.shape)
        
        # data_h2_act = np.clip(self.data_h2_act, 0., 1.)
        return self.data_h2_act
    # -------------------------------------------------------------------------
# =============================================================================
    

    

# ann = NumpyNeuralNetwork(hidden_unit=20, learning_rate=0.00001, epochs=1000, verbose=10)
# ann.fit(train_X, train_y, test_X, test_y)

# plt.plot(ann.loss_history_train)
# plt.plot(ann.loss_history_valid)
# plt.plot(train_y)
# plt.scatter(train_y.flatten(), ann.pred(train_X).flatten())
# plt.plot(train_y.flatten()-ann.pred(train_X).flatten())
    
# # N is batch size; D_in is input dimension;
# # H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 13, 15, 1

# # Create random input and output data
# x = train_X
# y = train_y

# # Randomly initialize weights
# w1 = np.random.rand(D_in, H)
# w2 = np.random.rand(H, D_out)

# learning_rate = 1e-6
# for t in range(1000):
#     # Forward pass: compute predicted y
#     h = x.dot(w1)
#     h_relu = np.maximum(h, 0)
#     y_pred = h_relu.dot(w2)

#     # Compute and print loss
#     loss = np.square(y_pred - y).sum()
#     if t % 999 == 0:
#         print(t, loss)

#     # Backprop to compute gradients of w1 and w2 with respect to loss
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_w2 = h_relu.T.dot(grad_y_pred)
#     grad_h_relu = grad_y_pred.dot(w2.T)
#     grad_h = grad_h_relu.copy()
#     grad_h[h < 0] = 0
#     grad_w1 = x.T.dot(grad_h)

#     # Update weights
#     w1 -= learning_rate * grad_w1
#     w2 -= learning_rate * grad_w2




class PytorchNeuralNetwork():
    def __init__(self, hidden_unit, learning_rate=0.0001, epochs=500, verbose=100):
        
        self.hidden_unit   = hidden_unit
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.verbose       = verbose
        
        self.activation_name = 'relu'
        
        self.loss_history_train = []
        self.loss_history_valid = []
        
        self.device = torch.device("cpu")
        # device = torch.device("cuda:0") # Uncomment this to run on GPU
        self.dtype  = torch.float
        
        return None
    # -------------------------------------------------------------------------
    def activation(self, x):
        
        if self.activation_name == 'relu':
            # relu 激活函数
            out = torch.clamp(x, min=0)
        elif self.activation_name == 'sigmoid':
            # sigmoid 激活函数
            out = 1 / (1 + torch.exp(-x))
        elif self.activation_name == 'tanh':
            # tanh 激活函数
            out = torch.tanh(x)
        
        return out
    # -------------------------------------------------------------------------
    # def initailize(self):
        
    #     return None
    # -------------------------------------------------------------------------
    def fit(self, train_X, train_y, valid_X=None, valid_y=None):
        
        train_X = torch.tensor(train_X, device=self.device, dtype=self.dtype)
        train_y = torch.tensor(train_y, device=self.device, dtype=self.dtype)
        if not(valid_X is None):
            valid_X = torch.tensor(valid_X, device=self.device, dtype=self.dtype)
            valid_y = torch.tensor(valid_y, device=self.device, dtype=self.dtype)
        
        assert train_y.ndim == 2
        # print(train_y.ndim)
        
        self.n_sample_train = train_X.shape[0]
        
        self.input_unit  = train_X.shape[1]
        self.output_unit = train_y.shape[1]
        # Randomly initialize weights
        # [0, 1]均匀分布初始化
        self.w1 = torch.rand((self.input_unit,  self.hidden_unit), device=self.device, dtype=self.dtype, requires_grad=True)
        self.w2 = torch.rand((self.hidden_unit, self.output_unit), device=self.device, dtype=self.dtype, requires_grad=True)
        
        self.b1 = torch.rand(self.hidden_unit, device=self.device, dtype=self.dtype, requires_grad=True)
        self.b2 = torch.rand(self.output_unit, device=self.device, dtype=self.dtype, requires_grad=True)
        
            
        # 训练过程
        for i in range(self.epochs):
            
            pred_y = self.pred(train_X)
            
            loss = torch.pow((pred_y - train_y), 2).sum()
            
            # 训练集 损失函数日志
            self.loss_history_train.append(loss.item())
            
            # 训练过程输出日志
            if self.verbose == 0:
                pass
            elif i % self.verbose == 0: 
                print('t:{0:>4d}, loss(mse):{1:>12.6f}'.format(i, loss.item()))
            
            loss.backward()
            
            with torch.no_grad():
                # print(i)
                self.w1 -= self.learning_rate * self.w1.grad
                self.b1 -= self.learning_rate * self.b1.grad
                self.w2 -= self.learning_rate * self.w2.grad
                self.b2 -= self.learning_rate * self.b2.grad
                # 
                
                # 或者
                # self.w1.data = self.w1.data - self.learning_rate * self.w1.grad.data
                # self.b1.data = self.b1.data - self.learning_rate * self.b1.grad.data
                # self.w2.data = self.w2.data - self.learning_rate * self.w2.grad.data
                # self.b2.data = self.b2.data - self.learning_rate * self.b2.grad.data
                
                # Manually zero the gradients after updating weights
                self.w1.grad.zero_()
                self.b1.grad.zero_()
                self.w2.grad.zero_()
                self.b2.grad.zero_()
            # 验证集 损失函数日志, 注意: 调用self.pred() 会更新相关参数
            if not(valid_X is None):
                self.loss_history_valid.append(torch.pow((self.pred(valid_X) - valid_y), 2).sum().item())
            
        return None
    # ------------------------------------------------------------------------- 
    def pred(self, data_X):
        
        if not(isinstance(data_X, torch.Tensor)):
            data_X = torch.tensor(data_X, device=self.device, dtype=self.dtype)
        
        # 隐含层 输出
        self.data_h1_linear = torch.add(torch.mm(data_X, self.w1), self.b1.reshape(1, -1))
        # print(self.data_h1_linear.shape, self.b1.reshape(1, -1).shape)
        self.data_h1_act    = self.activation(self.data_h1_linear)
        # print('h1_activation', self.data_h1_act.shape)
        
        # 输出层 输出
        self.data_h2_linear = torch.add(torch.mm(self.data_h1_act, self.w2), self.b2.reshape(1, -1))
        # print(self.data_h2_linear.shape, self.b2.reshape(1, -1).shape)
        self.data_h2_act    = self.activation(self.data_h2_linear)
        # print('h2_activation', self.data_h2_act.shape)
        
        # data_h2_act = np.clip(self.data_h2_act, 0., 1.)
        return self.data_h2_act
    # -------------------------------------------------------------------------
# =============================================================================






class TwoLayerNet(torch.nn.Module):
    def __init__(self, input_unit, hidden_unit, output_unit):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(TwoLayerNet, self).__init__()
        self.input_linear  = torch.nn.Linear(input_unit, hidden_unit)
        self.output_linear = torch.nn.Linear(hidden_unit, output_unit)
        self.activation    = torch.nn.ReLU()

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        
        out = self.input_linear(x)
        out = self.activation(out)
        out = self.output_linear(out)
        out = self.activation(out)
        
        return out
    # -------------------------------------------------------------------------
# =============================================================================




class PytorchModelNeuralNetwork():
    def __init__(self, hidden_unit, learning_rate=0.0001, epochs=500, verbose=100):
        
        self.hidden_unit   = hidden_unit
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.verbose       = verbose
        
        self.activation_name = 'relu'
        
        self.loss_history_train = []
        self.loss_history_valid = []
        
        self.device = torch.device("cpu")
        # device = torch.device("cuda:0") # Uncomment this to run on GPU
        self.dtype  = torch.float
        
        return None
    # -------------------------------------------------------------------------
    def fit(self, train_X, train_y, valid_X=None, valid_y=None):
        
        train_X = torch.tensor(train_X, device=self.device, dtype=self.dtype)
        train_y = torch.tensor(train_y, device=self.device, dtype=self.dtype)
        if not(valid_X is None):
            valid_X = torch.tensor(valid_X, device=self.device, dtype=self.dtype)
            valid_y = torch.tensor(valid_y, device=self.device, dtype=self.dtype)
        
        assert train_y.ndim == 2
        # print(train_y.ndim)
        
        self.n_sample_train = train_X.shape[0]
        
        self.input_unit  = train_X.shape[1]
        self.output_unit = train_y.shape[1]
        # Randomly initialize weights
        # [0, 1]均匀分布初始化
        
        model = TwoLayerNet(self.input_unit, self.hidden_unit, self.output_unit)
        
        criterion = torch.nn.MSELoss(reduction='sum')
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        
        # 训练过程
        for i in range(self.epochs):
            
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(train_X)
        
            # # Compute and print loss
            loss = criterion(y_pred, train_y)
            # print(type(loss))
            if self.verbose == 0:
                pass
            elif i % self.verbose == 0: 
                print('t:{0:>4d}, loss(mse):{1:>12.6f}'.format(i, loss.item()))
                
            # 训练集 损失函数日志
            self.loss_history_train.append(loss.item())
            
            
            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()
            
            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()
            # 验证集 损失函数日志, 注意: 调用self.pred() 会更新相关参数
            if not(valid_X is None):
                self.loss_history_valid.append(criterion(model(valid_X), valid_y).item())
        return None
    # ------------------------------------------------------------------------- 
# =============================================================================








