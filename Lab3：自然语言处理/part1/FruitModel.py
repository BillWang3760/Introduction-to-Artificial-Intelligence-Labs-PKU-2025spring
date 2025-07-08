import math
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *

class NullModel:
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:
    def __init__(self):
        self.dataset = traindataset() # 完整训练集，需较长加载时间
        #self.dataset = minitraindataset() # 用来调试的小训练集，仅用于检查代码语法正确性

        # 以下内容可根据需要自行修改，不修改也可以完成本题
        self.token_num = [{}, {}] # token在正负样本中出现次数
        self.V = 0 #语料库token数量
        self.pos_neg_num = [0, 0] # 正负样本数量
        self.count()

    def count(self):
        # TODO: YOUR CODE HERE
        # 提示：统计token分布不需要返回值
        vocabulary = set()
        for text, label in self.dataset:
            recorded = set()  # 若某个单词在一句话中多次出现，则不重复统计
            if label == 1:
                self.pos_neg_num[0] += 1
                for token in text:
                    vocabulary.add(token)
                    if token in self.token_num[0] and token not in recorded:
                        self.token_num[0][token] += 1
                        recorded.add(token)
                    elif token not in self.token_num[0]:
                        self.token_num[0][token] = 1
                        recorded.add(token)
            else:
                self.pos_neg_num[1] += 1
                for token in text:
                    vocabulary.add(token)
                    if token in self.token_num[1] and token not in recorded:
                        self.token_num[1][token] += 1
                        recorded.add(token)
                    elif token not in self.token_num[1]:
                        self.token_num[1][token] = 1
                        recorded.add(token)
        self.V = len(vocabulary)

    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 返回1或0代表当前句子分类为正/负样本
        alpha = 1  # 拉普拉斯平滑
        prob_pos = self.pos_neg_num[0] / (self.pos_neg_num[0] + self.pos_neg_num[1])
        prob_neg = self.pos_neg_num[1] / (self.pos_neg_num[0] + self.pos_neg_num[1])
        for token in text:
            if token in self.token_num[0]:
                prob_token_pos = (self.token_num[0][token] + alpha) / (self.pos_neg_num[0] + self.V * alpha)
            else:
                prob_token_pos = alpha / (self.pos_neg_num[0] + self.V * alpha)
            prob_pos *= prob_token_pos
            if token in self.token_num[1]:
                prob_token_neg = (self.token_num[1][token] + alpha) / (self.pos_neg_num[1] + self.V * alpha)
            else:
                prob_token_neg = alpha / (self.pos_neg_num[1] + self.V * alpha)
            prob_neg *= prob_token_neg
        if prob_pos > prob_neg:
            return 1
        else:
            return 0


def buildGraph(dim, num_classes, L): #dim: 输入一维向量长度, num_classes:分类数
    # 以下类均需要在BaseNode.py中实现
    # 也可自行修改模型结构
    nodes = [Attention(dim), relu(), LayerNorm((L, dim)), ResLinear(dim), relu(), LayerNorm((L, dim)), Mean(1), Linear(dim, num_classes), LogSoftmax(), NLLLoss(num_classes)]
    
    graph = Graph(nodes)
    return graph


save_path = "model/attention.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector
        
    def __call__(self, text, max_len=50):
        # TODO: YOUR CODE HERE
        # 利用self.emb将句子映射为一个二维向量（LxD），注意，同时需要修改训练代码中的网络维度部分
        # 默认长度L为50，特征维度D为100
        # 提示: 考虑句子如何对齐长度，且可能存在空句子情况（即所有单词均不在emd表内） 
        # 长度对齐：不足补零，超出截断
        # 单词不在emd表内：忽略，下一个！
        vector2D = np.zeros((max_len, 100))
        length = 0
        for token in text:
            if token in self.emb:
                vector2D[length] = self.emb[token]
                length += 1
            if length == max_len:
                break
        return vector2D


class AttentionModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()

    def __call__(self, text, max_len=50):
        X = self.embedding(text, max_len)
        X = np.expand_dims(X, 0)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=-1)
        return haty[0]


class QAModel():
    def __init__(self):
        self.document_list = get_document()

    def tf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回单词在文档中的频度
        # document变量结构请参考fruit.py中get_document()函数
        tokenized_document = document['document']  # List token化的文档
        n = 0                        # n：某个词在文档中出现的次数
        N = len(tokenized_document)  # N：文档中所有词出现的次数之和
        for token in tokenized_document:
            if token == word:
                n += 1
        if N == 0:
            tf = 0.0
        else:
            tf = float(n) / float(N)
        return tf

    def idf(self, word):
        # TODO: YOUR CODE HERE
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        # 注意python整除与整数除法的区别
        d = 0                        # d：语料库中出现某个词的文档数量
        D = len(self.document_list)  # D：语料库中包含的文档总数量
        for document in self.document_list:
            tokenized_document = document['document']
            if word in tokenized_document:
                d += 1
                continue
        if D == 0:
            idf = 0.0
        else:
            idf = np.log10(float(D) / (1.0 + float(d)))
        return idf

    def tfidf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回TF-IDF值
        tf = self.tf(word, document)
        idf = self.idf(word)
        tfidf = tf * idf
        return tfidf

    def __call__(self, query):
        query = tokenize(query) # 将问题token化
        # TODO: YOUR CODE HERE
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，可以参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数
        # Files should be ranked according to the sum of tf-idf values for any word in the query that also appears in the file.
        tfidf_of_documents = {}
        for index, document in enumerate(self.document_list):
            tfidf_of_documents[index] = 0
            for token in query:
                tfidf_of_documents[index] += self.tfidf(token, document)
        best_document_index = max(tfidf_of_documents, key=tfidf_of_documents.get)
        best_document = self.document_list[best_document_index]
        # Sentences should be ranked according to “matching word measure”:
        # namely, the sum of IDF values for any word in the query that also appears in the sentence.
        sentences = best_document['sentences']
        idf_of_sentences = {}
        for index, sentence in enumerate(sentences):
            idf_of_sentences[index] = 0
            for token in query:
                if token in sentence[0]:
                    idf_of_sentences[index] += self.idf(token)
        best_sentence_index = max(idf_of_sentences, key=idf_of_sentences.get)
        best_sentence = sentences[best_sentence_index][1]
        return best_sentence


modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "Attn": AttentionModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 4e-3   # 学习率
    wd1 = 1e-4  # L1正则化
    wd2 = 1e-5  # L2正则化
    batchsize = 256
    max_epoch = 15
    # lr = 4e-3  wd2 = 1e-3  batchsize = 256  max_epoch = 15  acc = 0.7550
    # lr = 5e-3  wd2 = 1e-5  batchsize = 64   max_epoch = 15  acc = 0.7500
    # lr = 1e-3  wd2 = 1e-5  batchsize = 64   max_epoch = 10  acc = 0.7250
    # lr = 4e-3  wd2 = 1e-5  batchsize = 256  max_epoch = 15  acc = 0.7700

    max_L = 50
    num_classes = 2
    feature_D = 100
    
    graph = buildGraph(feature_D, num_classes, max_L) # 维度可以自行修改

    # 训练
    # 完整训练集训练有点慢
    best_train_acc = 0
    dataloader = traindataset(shuffle=True) # 完整训练集
    #dataloader = minitraindataset(shuffle=True) # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        for text, label in dataloader:
            x = embedding(text, max_L)
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)
            Y.append(label)
            cnt += 1
            if cnt == batchsize:
                X = np.stack(X, 0)
                Y = np.concatenate(Y, 0)
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=-1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)