from typing import List
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from .Init import *

EPS = 1e-6

def shape(X):
    if isinstance(X, np.ndarray):  # isinstance()：用于检查一个对象是否是一个特定类型‌
        ret = "ndarray"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return f" {X.shape} "
    if isinstance(X, int):
        return "int"
    if isinstance(X, float):
        ret = "float"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return ret
    else:
        raise NotImplementedError(f"unsupported type {type(X)}")

class Node(object):
    def __init__(self, name, *params):
        self.grad = [] # 节点的梯度，self.grad[i]对应self.params[i]在反向传播时的梯度
        self.cache = [] # 节点保存的临时数据
        self.name = name # 节点的名字
        self.params = list(params) # 用于Linear节点中存储weight和bias参数使用

    def num_params(self):
        return len(self.params)

    def cal(self, X):
        '''
        计算函数值。请在其子类中完成具体实现。
        '''
        pass

    def backcal(self, grad):
        '''
        计算梯度。请在其子类中完成具体实现。
        '''
        pass

    def flush(self):
        '''
        初始化或刷新节点内部数据，包括梯度和缓存
        '''
        self.grad = []
        self.cache = []

    def forward(self, X, debug=False):
        '''
        正向传播。输入X，输出正向传播的计算结果。
        '''
        if debug:
            print(self.name, shape(X))
        ret = self.cal(X)
        if debug:
            print(shape(ret))
        return ret

    def backward(self, grad, debug=False):
        '''
        反向传播。输入grad（该grad为反向传播到该节点的梯度），输出反向传播到下一层的梯度。
        '''
        if debug:
            print(self.name, shape(grad))
        ret = self.backcal(grad)
        if debug:
            print(shape(ret))
        return ret
    
    def eval(self):
        pass

    def train(self):
        pass

class relu(Node):
    # input X: (*)，即可能是任意维度
    # output relu(X): (*)
    def __init__(self):
        super().__init__("relu")

    def cal(self, X):
        self.cache.append(X)
        return np.clip(X, 0, None)
        # np.clip()：将数组中的元素限制在指定的最小值和最大值之间，超出范围的元素会被截断（小于最小值的用最小值代替，大于最大值的用最大值代替）

    def backcal(self, grad):
        return np.multiply(grad, self.cache[-1] > 0) 

class sigmoid(Node):
    # input X: (*)，即可能是任意维度
    # output sigmoid(X): (*)
    # 参见class tanh(Node)
    def __init__(self):
        super().__init__("sigmoid")

    def cal(self, X):
        # TODO: YOUR CODE HERE
        ret = 1 / (1 + np.exp(-X))
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        return np.multiply(grad, np.multiply(self.cache[-1], 1 - self.cache[-1]))
    
class tanh(Node):
    # input X: (*)，即可能是任意维度
    # output tanh(X): (*)
    def __init__(self):
        super().__init__("tanh")

    def cal(self, X):
        ret = np.tanh(X)
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        return np.multiply(grad, np.multiply(1+self.cache[-1], 1-self.cache[-1]))

class Linear(Node):
    # input X: (*,d1)
    # param weight: (d1, d2)
    # param bias: (d2)
    # output Linear(X): (*, d2)
    def __init__(self, indim, outdim):
        """
        初始化
        @param indim: 输入维度
        @param outdim: 输出维度
        """
        weight = kaiming_uniform(indim, outdim)
        bias = zeros(outdim)
        super().__init__("linear", weight, bias)

    def cal(self, X):
        # TODO: YOUR CODE HERE
        # self.params = list(params)  # 用于Linear节点中存储weight和bias参数使用
        self.cache.append(X)
        return X @ self.params[0] + self.params[1]

    def backcal(self, grad):
        '''
        需要保存weight和bias的梯度，可以参考Node类和BatchNorm类
        '''
        # TODO: YOUR CODE HERE
        # 数学推导较为复杂，参见学习笔记：Lab2-Q3 数学推导
        X = self.cache[-1]
        weight_grad = X.T @ grad
        self.grad.append(weight_grad)
        bias_grad = grad.sum(axis=0)
        self.grad.append(bias_grad)
        X_grad = grad @ self.params[0].T
        return X_grad

class StdScaler(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-3
    def __init__(self, mean, std):
        super().__init__("StdScaler")
        self.mean = mean
        self.std = std

    def cal(self, X):
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        return X

    def backcal(self, grad):
        return grad/ (self.std + self.EPS)
    


class BatchNorm(Node):
    # 批归一化（Batch Normalization）层：用于在神经网络中标准化输入数据，以加速训练并提高模型性能
    # 数学推导较为复杂，具体原理推导参见学习笔记
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-8
    def __init__(self, indim, momentum: float = 0.9):
        super().__init__("batchnorm", ones((indim)), zeros(indim))
        # ones((indim))：gamma参数（可学习缩放因子），初始化全为1
        # zeros(indim)： beta参数（可学习偏移量），初始化全为0
        # 初始状态下，output = 1 * normalized_X + 0，即初始时不改变分布
        self.momentum = momentum  # 动量参数：用于控制历史统计量（均值/方差）与新批次的混合比例
        # new_mean = momentum * old_mean + (1 - momentum) * current_batch_mean
        # new_std = momentum * old_std + (1 - momentum) * current_batch_std
        # 较高的动量参数意味着更依赖历史统计量，较低的动量参数会让统计量更快适应新数据
        self.mean = None        # 均值
        self.std = None         # 标准差
        self.updatemean = True  # 控制是否更新均值和方差的标志
        # 训练模式(True)：更新统计量，评估模式(False)：固定统计量
        self.indim = indim      # 样本的特征维度，即每个样本的特征数

    def cal(self, X):
        if self.updatemean:
            tmean, tstd = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
            # axis = 0：指定沿着第0个维度计算均值/标准差，对于二维数组来说就是按列运算
            # keepdims = True：保持结果的维度与输入数组的维度一致，只是计算的维度被压缩成1
            # 例如：X = np.array([[1, 2, 3],
            #                    [4, 5, 6],
            #                    [7, 8, 9]])
            # tmean = [4, 5, 6]  tstd = [2.449, 2.449, 2.449]
            if self.mean is None or self.std is None:  # 初始化
                self.mean = tmean
                self.std = tstd
            else:  # 动量更新
                self.mean *= self.momentum
                self.mean += (1-self.momentum) * tmean
                self.std *= self.momentum
                self.std += (1-self.momentum) * tstd
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)  # 对输入X进行标准化
        self.cache.append(X.copy())
        X *= self.params[0]         # 缩放
        X += self.params[1]         # 平移
        return X

    def backcal(self, grad):
        X = self.cache[-1]
        # Y = gamma * normalized_X + beta
        # gamma:(indim, )  beta(indim, )  normalized_X:(batch_size, indim)  Y:(batch_size,indim)
        self.grad.append(np.multiply(X, grad).reshape(-1, self.indim).sum(axis=0))  # gamma_grad
        # reshape(-1, self.indim): 将逐元素相乘后的结果重新塑形为二维数组
        # 对于全连接层，输入形状为(batch_size, indim)，无需额外处理
        # 对于卷积层，输入形状为(batch_size, indim, height, width)，需将空间维度展平，例如：(32, 64, 28, 28) → (32*28*28, 64)
        self.grad.append(grad.reshape(-1, self.indim).sum(axis=0))  # beta_grad
        return (grad*self.params[0])/ (self.std + self.EPS)         # X_grad
    
    def eval(self):
        self.updatemean = False

    def train(self):
        self.updatemean = True


class Dropout(Node):
    '''
    input shape (*)
    output (*)
    '''
    def __init__(self, p: float = 0.1):
        super().__init__("dropout")
        assert 0<=p<=1, "p 是dropout 概率，必须在[0, 1]中"
        self.p = p           # 神经元被丢弃的概率
        self.dropout = True

    def cal(self, X):
        if self.dropout:
            X = X.copy()
            mask = np.random.rand(*X.shape) < self.p
            # np.random.rand(*X.shape)：生成一个与输入X形状相同的随机数矩阵，每个元素的值是[0,1)区间内均匀分布的随机数
            # 将生成的随机数矩阵与self.p（丢弃概率）逐元素比较，得到一个布尔矩阵mask，True表示该位置的神经元将被丢弃，False表示保留
            np.putmask(X, mask, 0)  # 将X中与【mask中为True的位置】相对应的位置的元素置零
            X = X * (1/(1-self.p))         # 乘以1/(1-self.p)，保持输出的期望值不变
            self.cache.append(mask)        # 保存掩码，用于反向传播时恢复梯度
        return X
    
    def backcal(self, grad):
        if self.dropout:
            grad = grad.copy()
            np.putmask(grad, self.cache[-1], 0)  # 将grad中与【mask中为True的位置】相对应的位置梯度置零
            grad = grad * (1/(1-self.p))   # 同样乘以1/(1-self.p)，与前向传播的缩放因子匹配
        return grad
    
    def eval(self):
        self.dropout=False

    def train(self):
        self.dropout=True


class Softmax(Node):
    # input X: (*)
    # output softmax(X): (*), softmax at 'dim'
    # 参见课后练习3：机器学习 T4
    def __init__(self, dim=-1):
        super().__init__("softmax")
        self.dim = dim  # 指定在哪个维度上计算Softmax，默认值为-1（表示在最后一个维度上计算）

    def cal(self, X):
        X = X - np.max(X, axis=self.dim, keepdims=True)  # 防止数值溢出
        expX = np.exp(X)
        ret = expX / expX.sum(axis=self.dim, keepdims=True)
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        softmaxX = self.cache[-1]
        grad_p = np.multiply(grad, softmaxX)
        return grad_p - np.multiply(grad_p.sum(axis=self.dim, keepdims=True), softmaxX)


class LogSoftmax(Node):
    # input X: (*)
    # output logsoftmax(X): (*), logsoftmax at 'dim'
    # 参见class Softmax(Node)、课后练习3：机器学习 T4
    def __init__(self, dim=-1):
        super().__init__("logsoftmax")
        self.dim = dim

    def cal(self, X):
        # TODO: YOUR CODE HERE
        X = X - np.max(X, axis=self.dim, keepdims=True)
        expX = np.exp(X)
        softmaxX = expX / expX.sum(axis=self.dim, keepdims=True)
        logSoftmaxX = np.log(softmaxX + EPS)
        self.cache.append(softmaxX)
        return logSoftmaxX

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        softmaxX = self.cache[-1]
        return grad - np.multiply(grad.sum(axis=self.dim, keepdims=True), softmaxX)

class NLLLoss(Node):
    '''
    negative log-likelihood 损失函数
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的log概率。  y：(*) 个整数类别标签
    # 输出：NLL损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("NLLLoss")
        self.y = y

    def cal(self, X):
        y = self.y
        self.cache.append(X)
        return - np.sum(
            np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1))
        # np.expand_dims(y, axis=-1)：将一维数组y转换为二维数组，形状从(n,)变为（n,1)
        # np.take_along_axis(X, indices, axis=-1):
        # indicies：二维数组，表示每个样本的真实标签的索引，形状为(n,1)
        # axis=-1：指定沿最后一个轴进行操作，对于二维数组来说就是按行运算
        # e.g.
        # X = np.array([[0.1, 0.2, 0.3],
        #               [0.4, 0.5, 0.6]])
        # y = np.array([2, 0])
        # np.expanded_dims(y, axis=-1) = [[2]
        #                                 [0]]
        # np.take_along_axis(X, y_expanded, axis=-1) = [[0.3]
        #                                               [0.4]]

    def backcal(self, grad):
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        # np.put_along_axis(arr, indices, value, axis):
        # arr: 目标数组，值将被放入其中
        # indices: 索引数组，指定要放置值的位置
        # value: 要放置的值
        # axis: 沿哪个轴进行操作
        # e.g.
        # ret = [[0, 0, 0],
        #        [0, 0, 0]]
        # indices = [[2],
        #            [0]]
        # np.put_along_axis(ret, indices, -1, axis=-1) = [[0, 0, -1],
        #                                                 [-1, 0, 0]]
        return grad * ret

class CrossEntropyLoss(Node):
    '''
    多分类交叉熵损失函数，不同于课上讲的二分类。它与NLLLoss的区别仅在于后者输入log概率，前者输入概率。
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的概率。  y：(*) 个整数类别标签
    # 输出：交叉熵损失

    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("CELoss")
        self.y = y

    def cal(self, X):
        # TODO: YOUR CODE HERE
        # 提示，可以对照NLLLoss的cal
        y = self.y
        self.cache.append(X)
        log_X = np.log(X + EPS)
        return - np.sum(
            np.take_along_axis(log_X, np.expand_dims(y, axis=-1), axis=-1))

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        # 提示，可以对照NLLLoss的backcal
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret / X


def im2col(X, kernel_size, stride, padding):
    """
    将4D输入张量转换为2D矩阵（展开为列）
    输入形状：(N, C, H, W)
    输出形状：(N*H_out*W_out, C*kernel_size*kernel_size)
    """
    N, C, H, W = X.shape
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                      mode='constant')
    # 使用stride_tricks创建滑动窗口视图
    shape = (N, C, H_out, W_out, kernel_size, kernel_size)
    strides = (X_padded.strides[0], X_padded.strides[1],
               stride * X_padded.strides[2], stride * X_padded.strides[3],
               X_padded.strides[2], X_padded.strides[3])

    windows = np.lib.stride_tricks.as_strided(X_padded,
                                              shape=shape,
                                              strides=strides)
    return windows.transpose(0, 2, 3, 1, 4, 5).reshape(N * H_out * W_out, -1)


def col2im(cols, X_shape, kernel_size, stride, padding):
    N, C, H, W = X_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    H_out = (H_padded - kernel_size) // stride + 1
    W_out = (W_padded - kernel_size) // stride + 1

    cols_reshaped = cols.reshape(N, H_out, W_out, C, kernel_size, kernel_size)
    cols_reshaped = cols_reshaped.transpose(0, 3, 1, 2, 4, 5)  # (N, C, H_out, W_out, K, K)

    grad_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    for i in range(kernel_size):
        for j in range(kernel_size):
            # 生成坐标网格 (直接创建二维坐标)
            h_idx = (np.arange(H_out) * stride + i).reshape(-1, 1)  # (H_out, 1)
            w_idx = (np.arange(W_out) * stride + j).reshape(1, -1)  # (1, W_out)
            # 广播为 (H_out, W_out) 的二维坐标
            h_grid = h_idx + np.zeros((1, W_out), dtype=int)  # (H_out, W_out)
            w_grid = w_idx + np.zeros((H_out, 1), dtype=int)  # (H_out, W_out)
            # 计算有效掩码
            valid = (h_grid < H_padded) & (w_grid < W_padded)
            # 获取有效位置的坐标（展平为一维）
            h_valid = h_grid[valid]
            w_valid = w_grid[valid]
            # 提取当前核位置对应的数据
            current_data = cols_reshaped[:, :, :, :, i, j]  # (N, C, H_out, W_out)
            # 将数据累加到对应位置
            if h_valid.size > 0:
                np.add.at(
                    grad_padded,
                    (slice(None), slice(None), h_valid, w_valid),
                    current_data[:, :, valid]  # (N, C, num_valid)
                )
    return grad_padded[:, :, padding:H + padding, padding:W + padding]

class Conv2d(Node):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        # Kaiming初始化权重
        scale = np.sqrt(2.0 / (in_channels * kernel_size ** 2))
        weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        bias = np.zeros(out_channels)
        super().__init__("conv2d", weight, bias)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

    def cal(self, X):
        # 保存原始输入形状
        self.cache.append(X.shape)
        N, _, H_in, W_in = X.shape
        H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        X_col = im2col(X, self.kernel_size, self.stride, self.padding)
        weight_col = self.params[0].reshape(self.out_channels, -1)
        output = X_col @ weight_col.T + self.params[1]
        return output.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)

    def backcal(self, grad):
        X_shape = self.cache.pop()
        N, C_out, H_out, W_out = grad.shape
        C_in = self.in_channels
        grad_reshaped = grad.transpose(0, 2, 3, 1).reshape(-1, C_out)
        # 计算权重梯度
        X_col = im2col(np.zeros(X_shape), self.kernel_size, self.stride, self.padding)  # 重建X_col结构
        weight_grad = grad_reshaped.T @ X_col
        self.grad.append(weight_grad.reshape(self.params[0].shape))
        # 计算偏置梯度
        self.grad.append(grad_reshaped.sum(axis=0))
        # 计算输入梯度
        weight_col = self.params[0].reshape(C_out, -1)
        grad_input_col = grad_reshaped @ weight_col
        return col2im(grad_input_col, X_shape, self.kernel_size, self.stride, self.padding)


class MaxPool2d(Node):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__("maxpool2d")
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def cal(self, X):
        N, C, H_in, W_in = X.shape
        H_out = (H_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        X_col = im2col(X, self.kernel_size, self.stride, self.padding)
        X_col = X_col.reshape(-1, self.kernel_size ** 2)
        self.max_idx = np.argmax(X_col, axis=1)
        self.cache.append(X.shape)
        return X_col[np.arange(X_col.shape[0]), self.max_idx].reshape(N, H_out, W_out, C).transpose(0, 3, 1, 2)

    def backcal(self, grad):
        X_shape = self.cache.pop()
        N, C, H_out, W_out = grad.shape
        grad_flat = grad.transpose(0, 2, 3, 1).ravel()
        grad_col = np.zeros((N * H_out * W_out * C, self.kernel_size ** 2))
        grad_col[np.arange(grad_col.shape[0]), self.max_idx] = grad_flat
        return col2im(grad_col, X_shape, self.kernel_size, self.stride, self.padding)


class Reshape(Node):
    def __init__(self, *target_shape):
        super().__init__("reshape")
        self.target_shape = target_shape

    def cal(self, X):
        self.cache.append(X.shape)
        return X.reshape(X.shape[0], *self.target_shape)

    def backcal(self, grad):
        return grad.reshape(self.cache.pop())
