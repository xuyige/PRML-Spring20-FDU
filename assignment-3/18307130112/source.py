import numpy as np
import random
from matplotlib import pyplot as plt

class GMM():
    def __init__(self, dataset, k, dims, length):
        self.k = k # 子高斯分布数
        self.mean = np.array([dataset[random.randint(0, len(dataset))] for i in range(k)]) #任取三点作为均值点初始值
        self.cov = np.full((k, dims, dims), np.diag(np.full(dims, 1.0))) #协方差
        self.p = np.ones(k)/k #先验概率
        self.dims= dims #高斯分布数据维数
        self.length = length #总数据数

    def EM(self, dataset, times):
        for i in range(times):
            posterior = np.zeros((self.length, self.k))
            P = np.zeros(self.k)
            for j in range(self.length):
                for l in range(self.k):
                    diff = dataset[j] - self.mean[l]
                    P[l] = self.p[l] * (np.exp(-0.5 * diff.T.dot(np.linalg.inv(self.cov[l])).dot(diff))) / (
                                pow(2 * np.pi, self.dims / 2) * pow(np.linalg.det(self.cov[l]), 0.5))
                posterior[j] = P / np.sum(P)
            nextp = np.sum(posterior, axis=0)
            self.p = nextp / self.length
            self.mean = np.array([1 / nextp[j] * np.sum(dataset * posterior[:, j].reshape((self.length, 1)), axis=0) for j in range(self.k)])
            for j in range(self.k):
                self.cov[j] = 0
                for l in range(self.length):
                    self.cov[j] += np.dot((dataset[l].reshape(1, self.dims) - self.mean[j]).T,(dataset[l] - self.mean[j]).reshape(1, self.dims)) * posterior[l, j]
                self.cov[j] = self.cov[j] / nextp[j]
            #print(i)
            if(i % 4 == 0):
                predict_k = np.argmax(posterior, axis = 1)
                plt.scatter(dataset[:, 0], dataset[:, 1], c = predict_k, marker = '.')
                print(i / 4 + 1)
                plt.subplot(5, 5, i / 4 + 1)

def gaussian_distribution(mean, cov, size):
    gauss1 = np.random.multivariate_normal(mean[0], cov[0], size[0])
    gauss2 = np.random.multivariate_normal(mean[1], cov[1], size[1])
    gauss3 = np.random.multivariate_normal(mean[2], cov[2], size[2])
    dataset = np.vstack((gauss1, gauss2, gauss3))
    dataset = np.array(dataset)
    labels = size[0] * [0] + size[1] * [1] + size[2] * [2]
    length = size[0] + size[1] + size[2]
    plt.plot(gauss1[:, 0], gauss1[:, 1], 'r.', label = 'label 1')
    plt.plot(gauss2[:, 0], gauss2[:, 1], 'g.', label = 'label 2')
    plt.plot(gauss3[:, 0], gauss3[:, 1], 'b.', label = 'label 3')
    plt.title("gaussian_distribution for each label")
    plt.show()
    randomset = random.sample(range(length), length)
    dataset = dataset[randomset]
    with open('dataset.data', 'w') as file:
        for i in range(length):
            file.write(str(dataset[i][0])+' '+str(dataset[i][1])+'\n')


def readfile():
    dataset= []
    with open('dataset.data', 'r') as file:
        for readline in file.readlines():
            readline = readline[0:-1].split(' ')
            dataset.append([float(readline[0]), float(readline[1])])
    dataset = np.array(dataset)
    return dataset


if __name__ == '__main__':
    #mean = np.array([[1, 0], [4, 0], [1, 4]])
    #cov = np.array([[[0.1, 0], [0, 0.5]],[[0.5, 0], [0, 0.5]],[[0.5, 0], [0, 0.5]]])
    mean = np.array([[1, 0], [2, 0], [0.3, 0.5]])
    cov = np.array([[[0.1, 0], [0, 0.1]]] * 3)
    size = np.array([170, 170, 170])
    gaussian_distribution(mean, cov, size)
    dataset = readfile()
    model = GMM(k=3, dims=2, dataset=dataset, length=np.sum(size))
    model.EM(dataset, 100)
    plt.show()
