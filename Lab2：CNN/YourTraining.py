import nni
import pickle
import mnist
import numpy as np
import scipy.ndimage as ndimage
from autograd.BaseGraph import Graph
from autograd.BaseNode import *
from autograd.utils import PermIterator
from util import setseed

# acc:0.88
# lr:0.0005302488960615706
# wd2:0.00001814335317919566
# batchsize:128

# acc: 0.88
# lr: 0.0010997728507944045
# wd2: 0.00003787381053179886
# batchsize: 256

def buildGraph(Y):
    nodes = [
        Reshape(1, 28, 28),

        Conv2d(1, 6, 5, 1, 2),   # Output：(1, 6, 28, 28)
        relu(),
        MaxPool2d(2,2, 0),  # Output：(1, 6, 14, 14)

        Conv2d(6, 16, 5, 1, 0),  # Output：(1, 16, 10, 10)
        relu(),
        MaxPool2d(2, 2, 0), # Output：(1, 16, 5, 5)

        Reshape(16 * 5 * 5),
        Linear(400, 120),
        BatchNorm(120),
        relu(),
        Linear(120, 84),
        relu(),

        Linear(84, 10),
        LogSoftmax(),
        NLLLoss(Y)
    ]
    graph = Graph(nodes)
    return graph

if __name__ == "__main__":

    save_path = "model/MyModel.npy"

    # 获取NNI参数
    params = nni.get_next_parameter()
    lr = params.get("lr", 1e-3)
    wd1 = 0
    wd2 = params.get("wd2", 1e-5)
    batchsize = params.get("batchsize", 128)

    # 数据加载与增强
    setseed(0)
    X = np.concatenate([mnist.trn_X, mnist.val_X], axis=0)
    Y = np.concatenate([mnist.trn_Y, mnist.val_Y], axis=0)
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
    index = np.random.choice(210000, size=70000, replace=False)
    X = X[index]
    Y = Y[index]

    # 构建模型
    graph = buildGraph(Y)

    # 训练循环
    best_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for epoch in range(1, 5 + 1):
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
        acc = np.average(np.concatenate(hatys) == np.concatenate(ys))
        print(f"epoch {epoch} loss {loss:.3e} acc {acc:.4f}")
        nni.report_intermediate_result(acc)  # 上报中间结果
        if acc > best_acc:
            best_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)

    # 验证并上报最终结果
    with open("model/MyModel.npy", "rb") as f:
        graph = pickle.load(f)
    graph.eval()
    test_X = mnist.test_X.reshape(-1, 28 * 28)
    pred = graph.forward(test_X, removelossnode=1)[-1]
    final_acc = np.average(np.argmax(pred, axis=1) == mnist.test_Y)
    nni.report_final_result(final_acc)  # 上报最终准确率
