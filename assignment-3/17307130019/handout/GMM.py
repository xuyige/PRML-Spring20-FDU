import os
import csv
os.sys.path.append('..')
# from handout import *
import math
import random
import numpy as np
import matplotlib.pyplot as plt

TINY = 0.0000001        # 防止特殊条件下分母为零，增加一个极小值

class GMM:
    def __init__(self, Data, K, weight=None, mean=None, cov=None):
        # Data -- 训练数据集
        # K -- 高斯分布个数
        # weight -- 高斯分布权重（初始构造概率）
        # mean -- 高斯分布均值向量（集合）
        # cov -- 高斯分布协方差矩阵（集合）

        self.Data = Data
        self.K = K
        N, dim = np.shape(self.Data)

        if weight is not None and len(weight) == K:
            self.weight = weight
        else:
            self.weight  = [1 / K] * K    # 默认或者参数错误，使用相同概率
        
        if mean is not None and len(mean) == dim:
            self.mean = mean
        else:
            self.mean = []
            for i in range(self.K):
                # t_mean = [0] * dim
                t_mean = np.random.rand(dim)
                # t_mean = t_mean / np.sum(t_mean)
                self.mean.append(t_mean)
        
        if cov is not None and np.shape(cov) == [dim, dim]:
            self.cov = cov
        else:
            self.cov  = []
            for i in range(self.K):
                # t_cov = [[0] * dim] * dim
                t_cov = np.random.rand(dim, dim)
                # t_cov = t_cov / np.sum(t_cov)
                self.cov.append(t_cov)

    def EM(self, threshold):
        N, dim = np.shape(self.Data)
        probs = [[0] * self.K] * N       # 第i个样本点属于第k个高斯混合模型的概率
        likelyhood = 1
        new_likelyhood = 0
        cnt = 0
        while abs(likelyhood - new_likelyhood) > threshold:
            print('Iteration ', cnt, ':', abs(likelyhood - new_likelyhood)) 
            
            likelyhood = new_likelyhood
            cnt += 1
            # E-step
            for n in range(N):
                post = [self.weight[k] * 
                        self.Gaussian(self.Data[n], self.mean[k], self.cov[k])
                        for k in range(self.K)]
                post = np.array(post)
                sum = np.sum(post)
                probs[n] = post / sum

            # M-step
            for k in range(self.K):
                N_k  = np.sum(probs[n][k] for n in range(N))
                # 更新各项参数
                self.weight[k] = 1.0 * N_k / N
                self .mean[k] = (1.0 / N_k) * np.sum([probs[n][k] * self.Data[n] 
                                for n in range(N)], axis=0)
                diff = self.Data - self.mean[k]
                self.cov[k] = (1.0 / N_k) * np.sum([probs[n][k] * 
                                diff[n].reshape((dim, 1)).dot(diff[n].reshape((1, dim)))
                                for n in range(N)], axis=0)
            
            new_likelyhood = []
            for n in range(N):
                t = [[np.sum(self.weight[k] * 
                    self.Gaussian(self.Data[n], self.mean[k], self.cov[k])) 
                    for k in range(self.K)]]
                for k in range(self.K):
                    if t[0][k] == 0: t[0][k] += TINY
                t = np.log(np.array(t))
                new_likelyhood.append(list(t))
            new_likelyhood = np.sum(new_likelyhood)
        for n in range(N):
            probs[n] = probs[n] / np.sum(probs[n])
        self.posibility = probs
        self.pred = [np.argmax(probs[n]) for n in range(N)]

    def Gaussian(self, x, mean, cov):
        dim = np.shape(cov)[0]
        cov_det = np.linalg.det(cov + np.eye(dim) * TINY)
        cov_inv = np.linalg.inv(cov + np.eye(dim) * TINY)
        diff = x - mean

        p = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(cov_det), 0.5)) *\
               np.exp(-0.5 * diff.dot(cov_inv).dot(diff.T))
        # print(p)
        return p

color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

def generate_data(N, K, p, para):
    label = [0] * N
    data = []
    for i in range(N):
        t = random.uniform(0, 1)
        for k in range(K):
            if t > np.sum(p[: k]) and t <= np.sum(p[: k + 1]):
                label[i] = k
                data.append(np.random.multivariate_normal(para[k][0], para[k][1:]))
    # print(data)
    data = np.array(data)
    # print(label)
    for i,x in enumerate(data):
        plt.scatter(x[0], x[1], c=color[label[i]])
    
    # plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    return label, data

A_DEFAULT = [[-1, -1], [0.5, 0], [0, 0.5]]
B_DEFAULT = [[2, 2], [2, 0.2], [0.2, 0.5]]
C_DEFAULT = [[-2, 3], [0.5, 0], [0, 0.5]]

def generate_gaussian_data(N, PA, PB, PC, A_para=A_DEFAULT, B_para=B_DEFAULT, C_para=C_DEFAULT):
    if PA + PB + PC != 1:
        print("Probability error.")
    A, B, C = [], [], []
    for i in range(N):
        t = random.uniform(0, 1)
        if t <= PA: 
            A.append(np.random.multivariate_normal(A_para[0], A_para[1:]))
        elif t <= PB: 
            B.append(np.random.multivariate_normal(B_para[0], B_para[1:]))
        else:
            C.append(np.random.multivariate_normal(C_para[0], C_para[1:]))

    res = np.concatenate([A, B, C], axis = 0)

    with open('dataset.data', 'w') as f:
        f.write(str(N) + "\n")
        for i in range(N):
            f.write(str(res[i]) + "\n")
    plt.plot(A[:, 0], A[:, 1], 'ro')
    plt.plot(B[:, 0], B[:, 1], 'bo')
    plt.plot(C[:, 0], C[:, 1], 'go')
    plt.show()



def process_gaussian_data():
    dic = []
    with open('dataset.data', 'r') as f:
        line = f.readline()
        N = int(line)
        for i in range(N):
            t = f.readline()
            t = t.split()
            dic.append([float(t[0]), float(t[0])])
    data_set = np.mat(dic)
    return data_set


def show_result(result):
    pass