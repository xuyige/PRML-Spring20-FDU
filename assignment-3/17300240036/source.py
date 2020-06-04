import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from scipy.stats import multivariate_normal

def generate_data(plot = True):
    num = [100, 300, 200]
    miu = [[0, 0], [-3, -3], [3, 3]]

    x = []
    for i in range(3):
        x.append(np.random.randn(num[i], 2) + miu[i])

    if plot:
        for i in range(3):
            plt.scatter(x[i][:, 0], x[i][:, 1])
        plt.show()
        plt.close()

    data = np.append(x[0], x[1], axis=0)
    data = np.append(data, x[2], axis=0)
    """
    data = np.append(data, np.zeros((np.sum(num), 1)), axis=1)
    
    sum = 0
    for i in range(3):
        data[sum:sum + num[i], 2] = i
        sum += num[i]
    """

    with open("data.data", "wb") as FILE:
        pickle.dump(data, FILE)

def load_data():
    with open("data.data", "rb") as FILE:
        return pickle.load(FILE)

class GMM():
    def __init__(self, k, dimension = 2):
        self.k = k
        self.dimension = dimension
        self.pi = 0
        self.mu = 0
        self.sigma = 0

    def calculate_gamma(self, data, pi, mu, sigma):
        n = len(data)
        gamma = np.zeros((n, self.k))
        for i in range(n):
            sum = 0
            for k in range(self.k):
                sum += pi[k] * multivariate_normal(mu[k], sigma[k]).pdf(data[i])
            for k in range(self.k):
                gamma[i][k] = pi[k] * multivariate_normal(mu[k], sigma[k]).pdf(data[i]) / sum
        return gamma
            
    def train(self, data, epoch):
        n = len(data)

        # pi 估计为n_k / n, 初始为1
        pi = np.ones(self.k)
        # mu 初始为随机挑选的数据点
        mu = np.array([data[random.randint(0, n)] for i in range(self.k)])
        #mu = np.array([[0, 0], [-3, -3], [3, 3]])
        # sigma 初始为0
        sigma = np.array([np.eye(self.dimension) for i in range(self.k)])

        for it in range(epoch):
            print(it)
            gamma = self.calculate_gamma(data, pi, mu, sigma)

            n_k = np.sum(gamma, axis=0) # n_k = sigma_i gamma_ik
            print(n_k)
            print(data[0])
            print(gamma[0])
            
            mu = np.zeros(mu.shape)
            for i in range(n):
                mu += np.dot(gamma[i].reshape(3, 1), data[i].reshape(1, 2))
            mu = mu / n_k.reshape(3, 1)

            sigma = np.zeros(sigma.shape)
            for k in range(self.k):
                for i in range(n):
                    sigma[k] += gamma[i][k] * np.dot((data[i] - mu[k]).reshape(2, 1), (data[i] - mu[k]).reshape(1, 2))
                sigma[k] /= n_k[k]
            
            pi = n_k / n

        self.pi = pi
        self.mu = mu
        self.sigma = sigma

    def predict(self, data):
        gamma = self.calculate_gamma(data, self.pi, self.mu, self.sigma)
        label = np.argmax(gamma, axis=1)
        return label

generate_data(plot = True)

data = load_data()

#print(data)

model = GMM(3, 2)
model.train(data, 200)
label = model.predict(data)

plt.scatter(data[:,0], data[:,1], c=label, marker='.')
plt.scatter(model.mu[:,0], model.mu[:,1], marker='x', color='red')
plt.show() 

