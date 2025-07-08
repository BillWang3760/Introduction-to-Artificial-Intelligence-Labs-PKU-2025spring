import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 10    # 学习率
wd = 1e-2  # l2正则化项系数
# 如果对学习率在1e-2到1e-1之间进行调参，那么valid acc只能接近但很难达到0.98

def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    return (X @ weight + bias)
    # A@B：对矩阵A，B进行矩阵乘法


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    n, d = X.shape
    haty = predict(X, weight, bias)

    loss = 0
    for i in range(n):
        if Y[i] * haty[i] > -1<<9:
            loss += (-1/n) * np.log(1 + np.exp(-Y[i] * haty[i]))
        else:
            loss += (-1/n) * Y[i] * haty[i] * np.log(1 + np.exp(Y[i] * haty[i]))
    loss += wd * np.sum(weight * weight)
    # 根据Y[i] * haty[i]的大小分情况处理，其目的为防止exp(很大的数字)导致溢出，下同
    # 将乘以(-1/n)置于求和过程中执行，其目的为防止累加过程中loss过大而溢出，下同

    gradientOfWeight = np.zeros(d,)
    for i in range(n):
        if Y[i] * haty[i] < 1<<9:
            gradientOfWeight += (-1/n) * X[i] * Y[i] / (1 + np.exp(Y[i] * haty[i]))
        else:
            gradientOfWeight += (-1/n) * X[i] * Y[i] * np.exp(-Y[i] * haty[i]) / (1 + np.exp(-Y[i] * haty[i]))
    gradientOfWeight += 2 * wd * weight
    weight -= lr * gradientOfWeight

    gradientOfBias = 0
    for i in range(n):
        if Y[i] * haty[i] < 1<<9:
            gradientOfBias += (-1/n) * Y[i] / (1 + np.exp(Y[i] * haty[i]))
        else:
            gradientOfBias += (-1/n) * Y[i] * np.exp(-Y[i] * haty[i]) / (1 + np.exp(-Y[i] * haty[i]))
    bias -= lr * gradientOfBias

    return (haty, loss, weight, bias)
