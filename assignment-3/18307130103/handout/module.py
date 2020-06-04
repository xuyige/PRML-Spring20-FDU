import math
import numpy as np
import random

from . import generate
from itertools import permutations

def L2(vec):
    res = 0
    for i in range(vec.shape[0]):
        res += vec[i] ** 2
    return(res)

def Normal(x, mu, sigma):
    sigma = np.asmatrix(sigma)
    delta = np.asmatrix(x - mu)
    num = - (1 / 2) * delta * sigma.I * delta.T
    return(1. / (2. * math.acos(-1.) * math.sqrt(np.linalg.det(sigma))) * math.exp(num))

def Argmax(a):
    mx = 0
    for i in range(a.shape[0]):
        if (a[i] > a[mx]):
            mx = i
    return(int(mx))

class KMeans:
    def __init__(self):
        self.m = np.zeros((0, 2))

    def Train(self, sample, classes = 3, epoch = 100):
        m = classes
        n = sample.shape[0]
        d = sample.shape[1]
        x = sample

        dis = np.zeros(n)
        self.m = np.asarray([x[random.randint(0, n - 1)]])
        for i in range(n):
            dis[i] = math.sqrt(L2(self.m[0] - x[i]))
        dis = dis / sum(dis)
        for i in range(m - 1):
            seed = random.random()
            for j in range(n):
                seed -= dis[j]
                if (seed <= 0):
                    self.m = np.append(self.m, np.asarray([x[j]]), 0)
                    break        

        for i in range(epoch):
            s = [[] for j in range(m)]
            for j in range(n):
                belong = 0
                for k in range(m):
                    if (L2(x[j] - self.m[k]) <= L2(x[j] - self.m[belong])):
                        belong = k
                s[belong].append(x[j])
            
            self.m = np.zeros((m, d))
            for j in range(m):
                for k in range(len(s[j])):
                    self.m[j] += s[j][k] / len(s[j])             

    def Mean(self):
        return(self.m)

    def Classify(self, sample):
        label = []
        n = sample.shape[0]
        for j in range(n):
            belong = 0
            for k in range(self.m.shape[0]):
                if (L2(sample[j] - self.m[k]) <= L2(sample[j] - self.m[belong])):
                    belong = k
            label.append(belong)
        return(np.asarray(label))
                

class GMM:
    def __init__(self):
        self.mu = np.zeros((0, 2))
        self.pi = np.zeros((0))
        self.sigma = np.zeros((0, 2, 2))

    def Init(self, sample, classes = 3, epoch = 10):
        n = sample.shape[0]
        if (epoch == 0):
            for i in range(classes):
                ii = random.randint(0, n - 1)
                self.mu = np.append(self.mu, np.asarray([sample[ii]]), 0)
            self.pi = np.asarray([1 / classes for i in range(classes)])
            self.sigma = np.asarray([[[1, 0], [0, 1]] for i in range(classes)])
        else:
            kmeans = KMeans()
            kmeans.Train(sample, classes=classes, epoch=epoch)
            self.mu = kmeans.Mean()    
            label = kmeans.Classify(sample)
            self.pi = np.asarray([1 / classes for i in range(classes)])
            for i in range(n):
                self.pi[label[i]] += 1.
            self.pi /= n
            self.sigma = np.asarray([[[1, 0], [0, 1]] for i in range(classes)])


    def Train(self, sample, gtl, classes = 3, epoch = 100, epochk = 10):
        m = classes
        n = sample.shape[0]
        d = sample.shape[1]
        x = sample

        self.Init(sample, classes=m, epoch=epochk)
    
        label = np.zeros((n))
        for i in range(epoch):
            gamma = np.zeros((n, m))
            for j in range(n):
                p = [self.pi[k] * Normal(x[j], self.mu[k], self.sigma[k]) for k in range(m)]
                gamma[j] = np.asarray(p / sum(p))

            N = np.zeros(m)
            for j in range(n):
                N = np.asarray([N[k] + gamma[j, k] for k in range(m)])

            self.pi = np.asarray([N[k] / np.sum(N) for k in range(m)])
            self.mu = np.zeros((m, d))
            for j in range(n):
                self.mu = np.asarray([self.mu[k] + gamma[j][k] * x[j] for k in range(m)])
            self.mu = np.asarray([self.mu[k] / N[k] for k in range(m)])
            self.sigma = np.zeros((m, d, d))
            for j in range(n):
                for k in range(m):
                    delta = np.asmatrix(x[j] - self.mu[k])
                    self.sigma[k] = self.sigma[k] + gamma[j][k] * (delta.T * delta)
            self.sigma = np.array([self.sigma[k] / N[k] for k in range(m)])
            
            if i % 10 == 0:
                label_ = self.Classify(sample)
                acc = Checker(label_, gtl)
                print('the epoch is ' + str(i) + ', the accuracy is ' + str(acc))

    def Classify(self, sample):
        label = []
        m = self.pi.shape[0]
        for i in range(sample.shape[0]):
           p = [self.pi[k] * Normal(sample[i], self.mu[k], self.sigma[k]) for k in range(m)] 
           label.append(Argmax(np.asarray(p)))
        return(np.asarray(label))

def Checker(y, t):
    m = max(y) + 1
    p = [i for i in range(m)]
    l = permutations(p)
    acc = 0.
    n = y.shape[0]
    for p in l:
        same = 0
        for i in range(n):
            if p[y[i]] == t[i]:
                same += 1
        if (acc < same / n):
            acc = same / n
    return(acc)