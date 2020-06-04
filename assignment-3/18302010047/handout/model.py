import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def dist(x, y, dim):
    sum = 0
    for i in range(dim):
        sum = sum + (x[i] - y[i]) * (x[i] - y[i])
    return sum ** 0.5

def Kmeans(data, K, dim):
    mean = np.random.rand(K, dim)
    for i in range(5): # epochs
        newMean = np.zeros((K, dim))
        cnt = np.zeros(K)
        for sample in data:
            q = np.argmin(np.array([dist(sample, mean[j], dim) for j in range(K)]))
            cnt[q] = cnt[q] + 1
            newMean[q] += np.array(sample)
        for j in range(K):
            mean[j] = newMean[j] / cnt[j]
    # print(mean)
    return mean

def calcPI(data, K, dim, mean):
    n = len(data)
    cnt = np.zeros(K)
    for sample in data:
        cnt[np.argmin(np.array([dist(sample, mean[i], dim) for i in range(K)]))] += 1
    cnt = cnt / n
    # print(cnt)
    return cnt

def classPlot(data, K, dim, mean):
    def trans(x):
        X = [sample[0] for sample in x]
        Y = [sample[1] for sample in x]
        return X, Y

    for t in range(K):
        X, Y = trans(list(filter(lambda x: 
                                    np.argmin(np.array([dist(x, mean[i], dim) for i in range(K)])) == t, data)))
        plt.scatter(X, Y, alpha=0.3)
    
    # plt.legend()
    # plt.show()

class GMM():
    def __init__(self, K=3, dim=2, data=None, init='kmeans'):
        self.K = K
        self.dim = dim
        self.data = data
        initSample = [random.randint(0, len(data)) for i in range(K)]
        # random init
        if init == 'random':
            self.mean = np.random.rand(K, dim)
        # choose a point
        elif init == 'point':
            self.mean = np.array([[data[p][0], data[p][1]] for p in initSample])
        # using kmeans to init
        elif init == 'kmeans':
            self.mean = Kmeans(data, K, dim)
        else:
            print('please use proper init method')
            exit(0)
        self.cov = np.array([np.eye(dim) for i in range(K)])
        self.PI = calcPI(data, K, dim, self.mean)
        self.E = None

    def expectation(self):
        n = len(self.data)
        self.E = np.zeros((n, self.K))
        for i in range(n):
            sum = 0
            for j in range(self.K):
                tmp = self.PI[j] * multivariate_normal.pdf(self.data[i], self.mean[j], self.cov[j])
                sum = sum + tmp
                self.E[i][j] = tmp 
            for j in range(self.K):
                self.E[i][j] = self.E[i][j] / sum
        
    def maximaize(self):
        n = len(self.data)
        sum = 0
        newMean = np.zeros((self.K, self.dim))
        newCov = np.zeros((self.K, self.dim, self.dim))
        newPI = np.zeros(self.K)
        for i in range(self.K):
            sum = 0
            for j in range(n):
                x = np.array(self.data[j])
                delta = x - self.mean[i]
                newMean[i] = np.add(newMean[i], self.E[j][i] * x)
                newCov[i] = np.add(newCov[i], self.E[j][i] * 
                                    np.matmul(np.reshape((x - self.mean[i]), (self.dim, -1)),
                                              np.reshape(x - self.mean[i], (1, self.dim))))
                sum = sum + self.E[j][i]

            self.mean[i] = newMean[i] / sum
            self.cov[i] = newCov[i] / sum
            self.PI[i] = sum / n   

    def train(self, epochs=50):
        for i in range(epochs):
            print(f'epoch {i}:')
            self.expectation()
            self.maximaize()
            # classPlot(self.data, self.K, self.dim, self.mean)
            # if i % 10 == 0:
            #     plt.savefig(f'epoch{i}.png', dpi=50)
            # plt.clf()
            print('gaussian mean')
            print(self.mean)
            # print(self.cov)
            # print(self.cov)
