import os
import sys
import math
import numpy as np
import operator
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from handout.gaussian_distrubution import *
# from gaussian_distrubution import *


def softmax(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()

# 生成每个集合的概率P，均值mean和方差sigma
def get_parameter(A, B, C):
    data = []
    data.append(A); data.append(B); data.append(C)

    P = np.zeros((3))
    mean = np.zeros((3, 2))
    sigma = np.zeros((3, 2, 2))
    for i in range(3):
        P[i] = len(data[i]) / (len(A) + len(B) + len(C))
        mean[i] = data[i].sum(axis=0) / len(data[i])
    for i in range(len(A) + len(B) + len(C)):
        if i < len(A): k = 0; j = i
        elif len(A) <= i and i < len(A) + len(B): k = 1; j = i - len(A)
        else: k = 2; j = i - len(A) - len(B)
        sigma[k] += np.dot((data[k][j] - mean[k]).T, (data[k][j] - mean[k]))
    for i in range(3):
        sigma[i] /= len(data[i])
    return P, mean, sigma

# 判别函数，利用贝叶斯定理计算后验
def judge(x, P, mean, sigma):
    prob = np.zeros((3))
    lab = ['A', 'B', 'C']
    sum = 0
    for i in range(3):
        prob[i] = stats.multivariate_normal(mean[i], sigma[i]).pdf(x) * P[i]
        sum += prob[i]
    prob /= sum
    max = -1; ret = 0
    for i in range(3):
        if max < prob[i]:
            max = prob[i]
            ret = i
    return lab[ret]

def LGM(A, B, C, T, NA, NB, NC, label):

    print("\nGenerative Model")

    r, c = [0, 0, 0], [0, 0, 0]
    lab = ['A', 'B', 'C']
    P, mean, sigma = get_parameter(A, B, C)
    for i in range(len(T)):
        t = judge(T[i], P, mean, sigma)
        for j in range(3):
            if t == lab[j]:
                c[j] += 1
                if label[i] == lab[j]:
                    r[j] += 1
    show_accuracy(r[0], r[1], r[2], NA, NB, NC, c[0], c[1], c[2])

if __name__=='__main__':
    # generate_gaussian_data(200, 150, 150)
    A, B, C, T, NA, NB, NC, label= process_gaussian_data()
    LGM(A, B, C, T, NA, NB, NC, label)