import numpy as np
import math
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def getClusterAccuracy(label, y_pre):
    cm = confusion_matrix(label, y_pre)

    indices = linear_sum_assignment(_make_cost_m(cm))
    indices = np.asarray(indices)
    indices = np.transpose(indices)
    js = [e[1] for e in sorted(indices, key=lambda x: x[0])]
    cm2 = cm[:, js]
    return np.trace(cm2) / np.sum(cm2)

def getNorm(mu, std, rho, n):
    """
    生成二维正态分布数据
      mu: 1*2 中心坐标
      std: 1*2 x y 两个维度上的标准差
      rho: 1*1 [0, 1] x y 的相关系数
      n: 数据量
    """
    mean = mu
    matrix = [[std[0] * std[0], rho * std[0] * std[1]],
             [rho * std[0] * std[1], std[0] * std[1]]]
    return np.random.multivariate_normal(mean, matrix, n)

def getDataset():
    """
    生成数据集
    """
    n = 100
    d1 = getNorm([1, 1], [2, 2], 0.6, n)
    d2 = getNorm([5, 11], [2, 1], 0.3, n)
    d3 = getNorm([6, 4], [1, 1], 0.5, n)
    dt = np.concatenate((d1, d2, d3), axis=0)
    label = [0 for i in range(n)]
    label.extend([1 for i in range(n)])
    label.extend([2 for i in range(n)])
    return dt, label

def getDataset2():
    """
    生成数据集
    """
    n = 100
    d1 = getNorm([1, 1], [2, 2], 0.6, n)
    d2 = getNorm([5, 11], [2, 1], 0.3, n)
    d3 = getNorm([9, 4], [1, 1], 0.5, n)
    dt = np.concatenate((d1, d2, d3), axis=0)
    label = [0 for i in range(n)]
    label.extend([1 for i in range(n)])
    label.extend([2 for i in range(n)])
    return dt, label

def getRandCovs(dim: int):
    """
    随机生成协方差矩阵
    """
    matrix = np.random.rand(dim, dim)
    stds = [math.sqrt(matrix[i][i]) for i in range(dim)]
    for i in range(dim):
        for j in range(i, dim):
            matrix[i][j] *= stds[i] * stds[j]
            matrix[j][i] = matrix[i][j]
#     matrix = np.random.rand(dim, dim)
    return matrix

def norm(x, mean, covs):
    """
    计算多维正态分布概率密度
    """
    dim = np.shape(mean)[0]
    shape = covs.shape
    assert(shape[0] == dim and shape[1] == dim)
    covs_det = np.linalg.det(covs)
    if (0 == covs_det):
        covs += np.eye(dim) * 0.001
        covs_det = np.linalg.det(covs)
    assert(0 != covs_det)
    covs_inv = np.linalg.inv(covs)
    diff = (x - mean).reshape((1, dim))
    p = np.exp(-0.5 * diff.dot(covs_inv).dot(diff.T))[0][0] / \
        (np.power(2 * np.pi, dim / 2) * np.power(np.abs(covs_det), 0.5))
    return p

def log_norm(x, mean, covs):
    """
    计算多维正态分布概率密度的对数值
    """
    dim = np.shape(mean)[0]
    covs_det = np.linalg.det(covs)
    if (0 == covs_det):
        covs += np.eye(dim) * 0.001
        covs_det = np.linalg.det(covs)
    assert(0 != covs_det)
    covs_inv = np.linalg.inv(covs)
    diff = x - mean
    return -0.5 * diff.dot(covs_inv).dot(diff.T) \
        - 0.5 * dim * np.log(2 * np.pi) - 0.5 * np.log(np.abs(covs_det))

class GMM:
    """
    高斯混合模型
    """
    def __init__(self, k, data=None, n=10):
        self.k = k
        if data is not None:
            self.fit(data, n=n)

    def fit(self, data=None, k=0, n=10):
        """
        多次拟合取最优解
        n: 重复拟合次数
        """
        if k > 0:
            self.k = k
        # 数据量 n 与数据维度 dim
        self.n, self.dim = np.shape(data)
        self.p = 0
        i = 0
        while i < n:
            p, weights, means, covs_list, gamma = self.fit_once(data)
            if None != p and not np.isfinite(p):
                print("error infinite p")
            if None != p and np.isfinite(p):
                i += 1
                print(f"第 {i} 轮拟合：\nloglikelihood: {p}")
                if 1 == i or self.p < p:
                    self.p, self.weights, self.means, self.covs_list, self.gamma = p, weights, means, covs_list, gamma
        self.pred = [np.argmax(self.gamma[i]) for i in range(self.n)]
        print(f"最终结果：\nloglikelihood: {self.p}")
        
    def fit_once(self, data):
        # 随机初始化协方差矩阵
        weights = np.random.rand(self.k)
        weights /= np.sum(weights)
        covs_list = [getRandCovs(self.dim) for i in range(self.k)]
        # 初始化均值 mu
        means = [np.random.rand(self.dim) for i in range(self.k)]
        
        p = 0
        prev_p = 1
        
        gamma = np.zeros((self.n, self.k)) #[ for i in range(self.n)]
        while np.abs(p - prev_p) > 1e-6:
            prev_p = p
            
            # E步
            for i in range(self.n):
                ls = []
                for k in range(self.k):
                    tmp = weights[k] * norm(data[i], means[k], covs_list[k])
                    if not np.isfinite(tmp):
                        return None, None, None, None, None
                    ls.append(tmp)
#                 pp = np.array([weights[k] * norm(data[i], means[k], covs_list[k])
#                          for k in range(self.k)])
                pp = np.array(ls)
                pp_sum = np.sum(pp)
                gamma[i] = pp / pp_sum
            # M步
            for k in range(self.k):
                n_k = np.sum(gamma[:,k])
                weights[k] = n_k / self.n
                means[k] = np.sum([gamma[i][k] * data[i] for i in range(self.n)], axis=0) / n_k
                diff = data - means[k]
                covs_list[k] = np.sum([gamma[i][k] * diff[i].reshape((self.dim, 1)).dot(diff[i].reshape((1, self.dim))) for i in range(self.n)], axis=0) / n_k
            # log likelyhood
            p = 0
            for i in range(self.n):
                s = 0
                for k in range(self.k):
                    tmp = weights[k] * norm(data[i], means[k], covs_list[k])
                    if not np.isfinite(tmp):
                        return None, None, None, None, None
                    s += tmp
                p += np.log(s)
                # p += np.sum([np.log(weights[k]) * log_norm(data[i], means[k], covs_list[k]) for k in range(self.k)])

        for i in range(self.n):
            gamma[i] = gamma[i] / np.sum(gamma[i])
        return p, weights, means, covs_list, gamma
            
    def _predict(self, data):
        """
        预测单个数据
        """
        return np.argmax([self.weights[k] * norm(data, self.means[k], self.covs_list[k]) for k in range(self.k)])
    
    def predict(self, data=None):
        if data is None:
            return self.pred
        return [self._predict(i) for i in data]

def run(data, label, k=3, n=5, normalize=False, show=False):
    """
    计算聚类结果
    """
    np.random.seed(0)
    if normalize:
        # 对数据进行预处理
        data = Normalizer().fit_transform(data)
    # 数据可视化
    plt.scatter(data[:,0],data[:,1],c = label)
    plt.title("Dataset")
    if show:
        plt.show()
    else:
        plt.savefig("dataset.png")
    # GMM模型
    gmm = GMM(k, data, n)
    y_pre = gmm.pred
    plt.scatter(data[:, 0], data[:, 1], c=y_pre)
    plt.title("GMM Result")
    if show:
        plt.show()
    else:
        plt.savefig("result.png")
    return y_pre


customDataset = True
if customDataset:
    data, label = getDataset()
else:
    iris = load_iris()
    label = np.array(iris.target)
    data = np.array(iris.data)
y_pre = run(data, label, normalize=(not customDataset))
print(f"Accuracy: {getClusterAccuracy(label, y_pre)}")
