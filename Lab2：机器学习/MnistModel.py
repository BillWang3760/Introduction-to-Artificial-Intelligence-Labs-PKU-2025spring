import numpy as np
import modelLogisticRegression as LR
import modelTree as Tree
import modelRandomForest as Forest
import modelSoftmaxRegression as SR
import modelMultiLayerPerceptron as MLP
import YourTraining as YT
import pickle

class NullModel:

    def __init__(self):
        pass

    def __call__(self, figure):
        return 0


class LRModel:
    def __init__(self) -> None:
        with open(LR.save_path, "rb") as f:
            self.weight, self.bias = pickle.load(f)

    def __call__(self, figure):
        pred = figure @self.weight + self.bias
        return 0 if pred > 0 else 1

class TreeModel:
    def __init__(self) -> None:
        with open(Tree.save_path, "rb") as f:
            self.root = pickle.load(f)
    
    def __call__(self, figure):
        return Tree.inferTree(self.root, Tree.discretize(figure.flatten()))


class ForestModel:
    def __init__(self) -> None:
        with open(Forest.save_path, "rb") as f:
            self.roots = pickle.load(f)
    
    def __call__(self, figure):
        return Forest.infertrees(self.roots, Forest.discretize(figure.flatten()))


class SRModel:
    def __init__(self) -> None:
        with open(SR.save_path, "rb") as f:
            graph = pickle.load(f)
        self.graph = graph
        self.graph.eval()

    def __call__(self, figure):
        self.graph.flush()
        pred = self.graph.forward(figure, removelossnode=True)[-1]
        return np.argmax(pred, axis=-1)
    
class MLPModel:
    def __init__(self) -> None:
        with open(MLP.save_path, "rb") as f:
            graph = pickle.load(f)
        self.graph = graph
        self.graph.eval()

    def __call__(self, figure):
        self.graph.flush()
        pred = self.graph.forward(figure, removelossnode=True)[-1]
        return np.argmax(pred, axis=-1)

class MyModel:
    def __init__(self) -> None:
        # 使用二进制模式打开模型保存路径
        with open(YT.save_path, 'rb') as f:
            # 使用pickle反序列化加载计算图对象
            graph = pickle.load(f)
        # 将加载的计算图对象保存到实例变量中
        self.graph = graph
        # 将模型设置为评估模式（关闭dropout等训练时特有的行为）
        self.graph.eval()

    def __call__(self, figure):  # figure: n*d 输入样本
        # 清空计算图中的中间数据，准备进行新的前向传播
        self.graph.flush()
        # 执行前向传播计算，返回前向传播的结果列表，取最后一个节点的输出作为预测结果
        pred = self.graph.forward(figure, removelossnode=True)[-1]
        # 对预测结果沿着最后一个维度（类别维度）取最大值索引，返回每个样本的预测类别
        return np.argmax(pred, axis=-1)

modeldict = {
    "Null": NullModel,
    "LR": LRModel,
    "Tree": TreeModel,
    "Forest": ForestModel,
    "SR": SRModel,
    "MLP": MLPModel,
    "Your": MyModel
}

