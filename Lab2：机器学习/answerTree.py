import numpy as np
from copy import deepcopy
from typing import List, Callable

EPS = 1e-6

# 超参数，分别为树的最大深度、熵的阈值、信息增益函数
# TODO: You can change or add the hyperparameters here
hyperparams = {
    "depth": 10,
    "purity_bound": 1e-2,
    "gainfunc": "gainratio"
    }

def entropy(Y: np.ndarray):
    """
    计算熵
    @param Y: (n,), 标签向量
    @return: 熵
    """
    # TODO: YOUR CODE HERE
    unique_label, label_counts = np.unique(Y, return_counts=True)
    # 使用np.unique函数找出标签向量Y中的唯一值unique_label，并统计每个唯一值出现的次数
    # 例如，若feat为[0,1,0,0,1]，则ufeat为[0,1]，featcnt为[3,2]，表示值0出现3次，值1出现2次
    label_probability = label_counts / Y.shape[0]
    H_D = -sum(label_probability * np.log2(label_probability + EPS))
    return H_D

def gain(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 信息增益
    """
    feat = X[:, idx]
    # 提取数据集X中所有样本的第idx个特征列
    # 例如，若X是形状为(100,5)的数组（100个样本，每个样本5个特征），idx=2时会提取出第3个特征的所有取值，得到一个长度为100的一维数组
    ufeat, featcnt = np.unique(feat, return_counts=True)
    # 使用np.unique函数找出特征列feat中的唯一值ufeat，并统计每个唯一值出现的次数
    # 例如，若feat为[0,1,0,0,1]，则ufeat为[0,1]，featcnt为[3,2]，表示值0出现3次，值1出现2次
    featp = featcnt / feat.shape[0]
    # feat.shape[0]是样本总数，将每个值的出现次数除以样本总数得到每个值的出现概率
    # 例如，若featcnt为[3,2]且样本数为5，则featp为[0.6,0.4],表示值0的概率为60%，值1的概率为40%
    ret = 0
    # TODO: YOUR CODE HERE
    # enumerate()会为数组中的每个元素生成一个索引-值对(索引，值)
    # 例如：ufeat = [10,20,30]  list(enumerate(ufeat)) = [(0,10),(1,20),(2,30)]
    for index, unique_feature in enumerate(ufeat):
        mask = (feat == unique_feature)
        # 创建布尔掩码（True/False)数组，标记当前样本的特征值是否等于unique_feature
        # 例如：feat = [0,1,1,0]，当前unique_feature = 1，则mask = [False,True,True,False]
        ret -= featp[index] * entropy(Y[mask])
    ret += entropy(Y)
    return ret


def gainratio(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益比
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 信息增益比
    """
    ret = gain(X, Y, idx) / (entropy(X[:, idx]) + EPS)
    return ret


def giniD(Y: np.ndarray):
    """
    计算基尼指数
    @param Y: (n,), 样本的label
    @return: 基尼指数
    """
    u, cnt = np.unique(Y, return_counts=True)
    p = cnt / Y.shape[0]
    return 1 - np.sum(np.multiply(p, p))


def negginiDA(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算负的基尼指数增益
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 负的基尼指数增益
    """
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    ret = 0
    # enumerate()会为数组中的每个元素生成一个索引-值对(索引，值)
    # 例如：ufeat = [10,20,30]  list(enumerate(ufeat)) = [(0,10),(1,20),(2,30)]
    for i, u in enumerate(ufeat):
        mask = (feat == u)
        # 创建布尔掩码（True/False)数组，标记当前样本的特征值是否等于u
        # 例如：feat = [0,1,1,0]，当前u=1，则mask = [False,True,True,False]
        ret -= featp[i] * giniD(Y[mask])
    ret += giniD(Y)  # 调整为正值，便于比较
    return ret


class Node:
    """
    决策树中使用的节点类
    """
    def __init__(self): 
        self.children = {}          # 子节点列表，其中key是特征的取值，value是子节点（Node）
        self.featidx: int = None    # 用于划分的特征
        self.label: int = None      # 叶节点的标签

    def isLeaf(self):
        """
        判断是否为叶节点
        @return: bool, 是否为叶节点
        """
        return len(self.children) == 0


def buildTree(X: np.ndarray, Y: np.ndarray, unused: List[int], depth: int, purity_bound: float, gainfunc: Callable, prefixstr=""):
    """
    递归构建决策树。
    @params X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @params Y: (n,), 样本的label
    @params unused: List of int, 未使用的特征索引
    @params depth: int, 树的当前深度
    @params purity_bound: float, 熵的阈值
    @params gainfunc: Callable, 信息增益函数
    @params prefixstr: str, 用于打印决策树结构
    @return: Node, 决策树的根节点
    """
    
    root = Node()
    u, ucnt = np.unique(Y, return_counts=True)
    root.label = u[np.argmax(ucnt)]
    # print(prefixstr, f"label {root.label} numbers {u} count {ucnt}") #可用于debug
    # 当达到终止条件时，返回叶节点
    # TODO: YOUR CODE HERE
    purity = entropy(Y)
    if purity <= purity_bound:
        return root

    gains = [gainfunc(X, Y, i) for i in unused]
    idx = np.argmax(gains)
    root.featidx = unused[idx]
    unused = deepcopy(unused)
    unused.pop(idx)
    feat = X[:, root.featidx]
    ufeat = np.unique(feat)
    # 按选择的属性划分样本集，递归构建决策树
    # 提示：可以使用prefixstr来打印决策树的结构
    # TODO: YOUR CODE HERE
    for unique_feature in ufeat:
        mask = (feat == unique_feature)
        # 创建布尔掩码（True/False)数组，标记当前样本的特征值是否等于unique_feature
        # 例如：feat = [0,1,1,0]，当前unique_feature = 1，则mask = [False,True,True,False]
        newX = X[mask]
        newY = Y[mask]
        childNode = buildTree(newX, newY, unused, depth, purity_bound, gainfunc, prefixstr)
        root.children[unique_feature] = childNode

    return root

def inferTree(root: Node, x: np.ndarray):
    """
    利用建好的决策树预测输入样本为哪个数字
    @param root: 当前推理节点
    @param x: d*1 单个输入样本
    @return: int 输入样本的预测值
    """
    if root.isLeaf():
        return root.label
    child = root.children.get(x[root.featidx], None)
    # x[root.featidx]：从当前样本x中提取特征的值
    # 在子节点字典root.children中查找该特征值对应的子节点，若不存在则返回None
    return root.label if child is None else inferTree(child, x)
    # 子节点不存在：直接返回当前节点的label（内部结点保存训练时该路径下样本的多数类别），处理缺失分支/未见特征值的情况
    # 子节点存在：递归调用inferTree处理子节点，继续向下遍历树
