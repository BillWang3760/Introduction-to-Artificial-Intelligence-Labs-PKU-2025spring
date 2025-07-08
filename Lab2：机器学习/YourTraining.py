import pickle
import mnist
import numpy as np
import scipy.ndimage as ndimage
from autograd.BaseGraph import Graph
from autograd.BaseNode import *
from autograd.utils import PermIterator
from util import setseed

setseed(0)  # 固定随机数种子以提高可复现性

save_path = "model/MyModel.npy"

# 超参数, 使用nni调参
lr = 0.003680982918962288
wd1 = 0
wd2 = 0.00004302581608324699
dropout_rate = 0.33424858362100096
batchsize = 256

# 将训练集trn和验证集val合并成一个更大的数据集
# trn_X, trn_Y: 训练集的图片与标签，每个标签是一个0-9的整数
# val_X, val_Y: 验证集的图片与标签
X = np.concatenate([mnist.trn_X, mnist.val_X], axis=0)
Y = np.concatenate([mnist.trn_Y, mnist.val_Y], axis=0)
# e.g.
# trn_X = np.random.rand(60000, 784)         # 60k训练样本
# val_X = np.random.rand(10000, 784)         # 10k验证样本
# X = np.concatenate([trn_X, val_X], axis=0) # 70k样本  axis=0：沿行的方向（纵向）拼接，即增加样本数量

# 图像数据增强
augmentedData = []
augmentedLabels = []
for index, img in enumerate(X):
    img = np.array(img).reshape(28, 28)
    # 旋转
    rotation = ndimage.rotate(img, np.random.uniform(-15, 15), reshape=False)  # reshape=False: 在旋转图像时不改变图像的形状
    augmentedData.append(rotation.reshape(-1))
    augmentedLabels.append(Y[index])
    # 平移
    translation = ndimage.shift(img, [np.random.randint(-5, 5), np.random.randint(-5, 5)])
    # 将图像img随机平移，水平和垂直方向平移量均在-10到9各像素之间
    augmentedData.append(translation.reshape(-1))
    augmentedLabels.append(Y[index])
X = np.concatenate([X, np.array(augmentedData)], axis=0)
Y = np.concatenate([Y, np.array(augmentedLabels)], axis=0)

def buildGraph(Y):
    nodes = [
        BatchNorm(784),
        # 此处不使用StdScaler(mnist.mean_X, mnist.std_X)的原因：
        # StdScaler只能调用已经在mnist.py里计算好的平均值和标准差，节点中的cal本身并不能计算平均值和标准差
        # 但数据增强后mnist.mean_X和mnist.std_X已经和增强后的图片没有任何关系
        # 亦即：使用不相干的平均值和标准差对增强后的图片进行处理，导致图片中的特征被打乱，模型无法学到有效特征而持续处于欠拟合状态

        Linear(784, 512),
        relu(),
        Dropout(dropout_rate),
        BatchNorm(512),

        Linear(512, 256),
        relu(),
        Dropout(dropout_rate),
        BatchNorm(256),

        Linear(256, 128),
        relu(),
        Dropout(dropout_rate),
        BatchNorm(128),

        Linear(128, 64),
        relu(),
        Dropout(dropout_rate),
        BatchNorm(64),

        Linear(64, 32),
        relu(),
        Dropout(dropout_rate),
        BatchNorm(32),

        Linear(32, 10),
        LogSoftmax(),
        NLLLoss(Y)
    ]
    graph = Graph(nodes)
    return graph

if __name__ == "__main__":
    graph = buildGraph(Y)
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, 15+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)
    # # 测试
    # with open(save_path, "rb") as f:
    #     graph = pickle.load(f)
    # graph.eval()
    # graph.flush()
    # test_X = mnist.test_X.reshape(-1, 28*28)
    # pred = graph.forward(test_X, removelossnode=1)[-1]
    # haty = np.argmax(pred, axis=1)
    # print("valid acc", np.average(haty==mnist.test_Y))
