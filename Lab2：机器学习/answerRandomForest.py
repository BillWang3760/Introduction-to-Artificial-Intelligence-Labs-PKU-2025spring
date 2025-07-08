from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 25      # 树的数量
ratio_data = 0.5   # 采样的数据比例
ratio_feat = 0.5   # 采样的特征比例
hyperparams = {
    "depth":10,
    "purity_bound":1e-2,
    "gainfunc": gainratio
    } # 每颗树的超参数
depth = hyperparams["depth"]
purity_bound = hyperparams["purity_bound"]
gainfunc = hyperparams["gainfunc"]

def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    indexOfSamples = np.arange(X.shape[0])   # np.array([0,1,...,n-1])
    indexOfFeatures = np.arange(X.shape[1])  # np.array([0,1,...,d-1])
    listOfDecisionTrees = []
    for i in range(num_tree):
        # 样本扰动：对于每一个决策树，对训练集进行随机采样得到一个独立的训练集
        reducedIndexOfSamples = np.random.choice(indexOfSamples, size = int(X.shape[0] * ratio_data), replace = False)
        mask = np.isin(indexOfSamples, reducedIndexOfSamples)
        # np.isin(A, B)会检查数组A中的每个元素是否在数组B中
        # 返回一个与A形状相同的布尔数组，其中元素为True表示A中的元素在B中存在，否则为 False
        reducedX = X[mask]
        reducedY = Y[mask]
        # 属性扰动：对于每一个决策树，只选择数据集中的一部分样本特征进行子树子树划分训练
        reducedIndexOfFeatures = np.random.choice(indexOfFeatures, size = int(X.shape[1] * ratio_feat), replace = False)
        unused = reducedIndexOfFeatures.tolist()  # 将Numpy数组转换为列表
        decisionTree = buildTree(reducedX, reducedY, unused, depth, purity_bound, gainfunc)
        listOfDecisionTrees.append(decisionTree)
    return listOfDecisionTrees

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))  # 过滤掉无效预测
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
