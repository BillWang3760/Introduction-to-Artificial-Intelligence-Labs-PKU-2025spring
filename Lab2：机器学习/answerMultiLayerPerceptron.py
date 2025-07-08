import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
# TODO: You can change the hyperparameters here
lr = 4e-3   # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 1e-5  # L2正则化
batchsize = 128

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    # 参见answerSoftmaxRegression.py中的def buildGraph(Y)
    nodes = [
             StdScaler(mnist.mean_X, mnist.std_X),

             Linear(mnist.num_feat, 1024),
             relu(),
             Dropout(0.5),
             BatchNorm(1024),

             Linear(1024, 1024),
             relu(),
             Dropout(0.5),
             BatchNorm(1024),

             Linear(1024, 1024),
             relu(),
             Dropout(0.5),
             BatchNorm(1024),

             Linear(1024, 1024),
             relu(),
             Dropout(0.5),
             BatchNorm(1024),

             Linear(1024, mnist.num_class),
             LogSoftmax(),
             NLLLoss(Y)
    ]
    graph = Graph(nodes)
    return graph
