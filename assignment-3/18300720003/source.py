# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:35:27 2020

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
from scipy import stats, integrate
import seaborn as sns


def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)

def getExpectation(Y, mu, cov, alpha):
    N = Y.shape[0]
    K = alpha.shape[0]
    gamma = np.mat(np.zeros((N, K)))

    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)

    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma

def maximize(Y, gamma):
    N, D = Y.shape
    K = gamma.shape[1]
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)
    for k in range(K):
        Nk = np.sum(gamma[:, k])
        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha

def scale_data(Y):
    # 对每一维特征分别进行缩放
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    return Y
covlog=[]
mulog=[]
def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    return mu, cov, alpha
def GMM_EM(Y, K, times):
    #Y = scale_data(Y)
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
        mulog.append(mu)
        covlog.append(cov)
    return mu, cov, alpha

Y = np.loadtxt("./handout/sample.data")
matY = np.matrix(Y, copy=True)

K = 2

mu, cov, alpha = GMM_EM(matY, K, 100)

N = Y.shape[0]
gamma = getExpectation(matY, mu, cov, alpha)
category = gamma.argmax(axis=1).flatten().tolist()[0]
class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])

plt.plot(class1[:, 0], class1[:, 1], 'bo', label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'rs', label="class2")
plt.legend(loc="best")
plt.title("GMM Clustering By EM Algorithm")
plt.show()


sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))

for i in range(100):
    if(i%20==0):
        mean1=np.array(mulog)[i,0]
        mean2=np.array(mulog)[i,1]
        cov1=np.array(covlog)[i,0]
        cov2=np.array(covlog)[i,1]
    
        data1 = np.random.multivariate_normal(mean1, cov1, 200) 
        data2 = np.random.multivariate_normal(mean2, cov2, 200) 
        df1 = pd.DataFrame(data1, columns=["x", "y"])
        df2 = pd.DataFrame(data2, columns=["x", "y"])
        plt.figure()
        plt.xlabel("X")
        plt.ylabel("Y")
        x1, y1 = data1.T
        x2,y2=data2.T
        z1=np.zeros(200)
        plt.plot(x1,y1,'o',x2,y2,'.')
        plt.show()
