import os
import csv
os.sys.path.append('..')
from handout.GMM import *
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
'''
PARA = [[[-1, -2], [0.5, 0], [0, 0.5]],
        [[2, 2], [2, 0.2], [0.2, 0.5]],
        [[-2, 3], [0.5, 0], [0, 0.5]]]
'''

PARA = [[[-4, -4], [0.5, 0], [0, 0.5]],
        [[3, 5], [2, 0.2], [0.2, 0.5]],
        [[-4, 3], [0.5, 0], [0, 0.5]],
        [[0, 0], [0.5, 0], [0, 0.5]],
        [[1, -4], [0.5, 0], [0, 0.5]],
        [[3, -1], [0.5, 0], [0, 0.5]],]

'''
PARA = [[[0, 0], [0.1, 0.2], [0.2, 2]],
        [[0, 0], [2, 0.2], [0.2, 0.1]]]
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--K', type=int, default=3, help='The number of classes')
    parser.add_argument('-n', '--NUM', type=int, default=300, help='The number of data')
    parser.add_argument('-p', '--POSSIBILITY', type=bool, default=False, help='The number of data')
    ARGS = parser.parse_args()
    K = ARGS.K
    N = ARGS.NUM
    if ARGS.POSSIBILITY:
        print("Enter", K, "numbers as possibility:")
        p = [float(n) for n in input().split()]
        s = sum(p)
        for i in range(len(p)):
            p[i] = p[i] / s
        p = np.array(p)
    else: 
        p = np.random.dirichlet(np.ones(K), size=1)
        p = p[0]
    print(p)
    para = PARA
    label, data = generate_data(N, K, p, para)

    PARA = np.array(PARA)

    # gmm = GMM(data, K, p, PARA[:, 0], PARA[:, 1:])
    gmm = GMM(data, K, p)
    gmm.EM(0.000001)
    y_pred = gmm.pred
    # print("GMM predicting result：\n",y_pred)

    plt.scatter(data[:, 0], data[:, 1], c=y_pred)
    plt.title("Show GMM predictin.")
    plt.show()

