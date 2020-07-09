import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def getNorm(mu, std, rho, n):
    # 生成二维正态分布数据
    # mu: 1*2 中心坐标
    # std: 1*2 x y 两个维度上的标准差
    # rho: 1*1 [0, 1] x y 的相关系数
    # n: 数据量
    mean = mu
    matrix = [[std[0] * std[0], rho * std[0] * std[1]],
             [rho * std[0] * std[1], std[0] * std[1]]]
    return np.random.multivariate_normal(mean, matrix, n)

def getDataset():
    """
    生成数据集
    """
    n = [100, 200, 150]
    d1 = getNorm([0, 0], [1, 1], 0, n[0])
    d2 = getNorm([0, 2], [1, 1], 0, n[1])
    d3 = getNorm([0, 4], [1, 1], 0, n[2])
    dt = np.concatenate((d1, d2, d3), axis=0)
    label = [0 for i in range(n[0])]
    label.extend([1 for i in range(n[1])])
    label.extend([2 for i in range(n[2])])
    return dt, label

def plot(data, label, title, file=None):
    plt.scatter(data[:,0], data[:,1], c = label)
    plt.title(title)
    if None == file:
        plt.show()
    else:
        plt.savefig(file)

class GenerativeModel:
    def __init__(self, dim, n_classes):
        """
        dim: 数据维度
        n_classes: 分类数量
        """
        self.dim = dim
        self.n = n_classes
        self.means = [np.zeros(dim) for i in range(n_classes)]
        self.covs = [np.zeros((dim, dim)) for i in range(n_classes)]
        # 不同类的先验概率
        self.weights = [0 for i in range(n_classes)]
        
    def fit(self, data, labels):
        """
        拟合、训练数据
        计算各个类的数据的正态分布的均值 means 和 协方差矩阵 covs
        data 最后一个维度为分类
        """
        means = [np.zeros(self.dim) for i in range(self.n)]
        covs = [np.zeros((self.dim, self.dim)) for i in range(self.n)]
        count = [0 for i in range(self.n)]
        weights = [0 for i in range(self.n)]
        for t, i in enumerate(data):
            label = labels[t]
            count[label] += 1
            means[label] += i
        ndata = sum(count)
        for i in range(self.n):
            means[i] /= count[i]
            weights[i] = count[i] / ndata
        for t, i in enumerate(data):
            label = labels[t]
            delta = i - means[label]
            covs[label] += np.dot(delta.reshape(self.dim, 1), delta.reshape(1, self.dim))
        for i in range(self.n):
            covs[i] /= count[i]
        self.means, self.covs, self.weights = means, covs, weights

    def _predict(self, data):
        """
        预测单个数据
        """
        p = [stats.multivariate_normal(self.means[i], self.covs[i]).pdf(data) * self.weights[i] for i in range(self.n)]
        return np.argmax(p)
    
    def predict(self, data):
        result = []
        for i in data:
            result.append(self._predict(i))
        return result

def softmax(x):
    x = x.T
    x = x - np.max(x, axis=1)
    return np.exp(x) / np.sum(np.exp(x), axis=1)

class DiscriminativeModel:
    def __init__(self, dim, n_classes):
        self.dim = dim
        self.n = n_classes
    
    def fit(self, data, labels, epochs=200, lr=0.01):
        w = np.random.randn(self.dim + 1, self.n)
        
        for epoch in range(epochs):
            for i, j in enumerate(data):
                label = labels[i]
                x = np.append(j, 1).reshape(self.dim + 1, 1)
                y = softmax(np.dot(w.T, x)).reshape(1, self.n)
                # pred = np.argmax(y)
                t = np.eye(self.n)[label]
                w += lr * np.dot(x, t - y)
        self.weights = w
    
    def _predict(self, data):
        """
        预测单个数据
        """
        x = np.append(data, 1).reshape(self.dim + 1, 1)
        y = softmax(np.dot(self.weights.T, x)).reshape(1, self.n)
        return np.argmax(y)
    
    def predict(self, data):
        return [self._predict(i) for i in data]
    
dt, label = getDataset()
x_train, x_test, y_train, y_test = train_test_split(dt, label, test_size=0.3, random_state=0)
plot(x_test, y_test, "Test Data and Label", "test-dataset.png")

gm = GenerativeModel(2, 3)
gm.fit(x_train, y_train)
y_pred = gm.predict(x_test)
print("GenerativeModel Accuracy: ", accuracy_score(y_test, y_pred))
plot(x_test, y_pred, "Generative Model Predicts Labels", "GenerativeModel.png")

dm = DiscriminativeModel(2, 3)
dm.fit(x_train, y_train)
y_pred = dm.predict(x_test)
print("DiscriminativeModel Accuracy: ", accuracy_score(y_test, y_pred))
plot(x_test, y_pred, "Discriminative Model Predicts Labels", "DiscriminativeModel.png")