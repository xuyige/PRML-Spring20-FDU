import numpy as np
import matplotlib.pyplot as plt
from .K_means import KMeans


class GMM:

    def __init__(self, k, dim, max_iter=None):
        self.k = k
        self.dim = dim
        self.mu = np.random.rand(k, dim)
        self.sigma = np.array([np.eye(dim)] * k) * np.random.rand(k, 1, 1)
        self.pi = np.array([1 / k] * k)
        self.ll = 0
        self.max_iter = max_iter

    def fit(self, data, pre_train=False, plot=False):
        print("start GMM fitting.")

        if pre_train:
            kmeans = KMeans(self.k)
            cluster = kmeans.fit(data)
            clustered_data = [data[np.where(cluster == i)] for i in range(self.k)]
            self.mu = np.array([np.mean(dat, axis=0) for dat in clustered_data])
            self.sigma = np.array([np.cov(dat.T) for dat in clustered_data])
            self.pi = np.array([dat.shape[0] / data.shape[0] for dat in clustered_data])

        def EM():
            count = 0
            while True:
                if count % 50 == 0:
                    print("EM iteration", count)
                prob = self.prob(data)
                sp = np.sum(prob, axis=1, keepdims=True)
                ll = np.sum(np.log(sp))
                if abs(ll - self.ll) < 1e-4 or (self.max_iter and count > self.max_iter):
                    break
                self.ll = ll
                res = prob / sp
                n = np.sum(res, axis=0)
                for i in range(self.k):
                    resp = res[:, i].reshape(data.shape[0], 1)
                    self.mu[i] = np.sum(resp * data, axis=0) / n[i]
                    self.sigma[i] = np.matmul((data - self.mu[i]).T, resp * (data - self.mu[i])) / n[i]
                self.pi = n / data.shape[0]
                count += 1

            print("End EM iteration.")
            return self.prob(data)

        clusters = np.argmax(EM(), axis=1)
        if plot and self.dim == 2:
            plt.scatter(data[:, 0], data[:, 1], c=clusters)
            plt.scatter(self.mu[:, 0], self.mu[:, 1], c='r')
            plt.show()

        return clusters

    def prob(self, data):
        def func(x):
            p = self.pi * np.array([self.norm_pdf(x, self.mu[i], self.sigma[i]) for i in range(self.k)])
            return p

        return np.apply_along_axis(func, 1, data)

    @staticmethod
    def norm_pdf(x, mu, sigma):
        return np.exp(-np.matmul(np.matmul(x - mu, np.linalg.inv(sigma)), (x - mu).T) / 2) / \
               ((2 * np.pi) ** (x.shape[0] / 2) * np.sqrt(np.linalg.det(sigma)))
