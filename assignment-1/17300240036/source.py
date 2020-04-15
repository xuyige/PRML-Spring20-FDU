import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import sys
from scipy.stats import multivariate_normal

def generate_data(plot = True):
    num = [300, 300, 300]
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
    data = np.append(data, np.zeros((np.sum(num), 1)), axis=1)
    
    sum = 0
    for i in range(3):
        data[sum:sum + num[i], 2] = i
        sum += num[i]

    with open("data.data", "wb") as FILE:
        pickle.dump(data, FILE)

def load_data():
    with open("data.data", "rb") as FILE:
        return pickle.load(FILE)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class discriminative_model:

    def __init__(self):
        self.w = np.zeros((3, 3))

    def model(self, X, Theta):
        return sigmoid(X @ Theta.T)

    """
    def cost(self, X, y, Theta):
        left = -y * np.log(model(X, Theta))
        right = (1 - y) * np.log(1 - model(X, Theta))
        return np.sum(right - left, axis=0) / len(X)
    """

    def gradient(self, X, y, Theta):
        grad = np.zeros(Theta.shape)
        error = (self.model(X, Theta) - y)
        for i in range(3):
            term = error[:, i:i+1] * X
            grad[i] = np.sum(term, axis=0) / len(X)
        return grad

    def construct_matrix(self, data):
        #print(data)
        X = data[:, 0:2]
        X = np.append(X, np.ones((len(X), 1)), axis=1)
        label = data[:, 2]
        y = np.zeros((len(X), 3))
        y[np.arange(len(X)), label.astype(np.int16)] = 1
        return X, y


    def train(self, data, epoch, batch_size, learning_rate):
        iteration = len(data) // batch_size
        for ep in range(epoch):
            np.random.shuffle(data)
            #print(data)
            #print("...........................")
            for i in range(iteration):
                #print(data[i * batch_size : (i + 1) * batch_size])
                X, y = self.construct_matrix(data[i * batch_size : (i + 1) * batch_size])
                #print(X, y)
                #sys.exit(0)
                self.w -= learning_rate * self.gradient(X, y, self.w)
                """
                print(X, y)
                print(self.w)
                print(self.model(X, self.w))
                """

    def predict(self, data, plot = True):
        X, label = self.construct_matrix(data)
        label = data[:, 2]
        y = self.model(X, self.w)
        #print(y)

        color = ['r', 'g', 'b']

        correct = 0
        for i in range(len(X)):
            output = np.argmax(y[i])
            if(output == label[i]):
                plt.plot(X[i][0], X[i][1], color[output] + 'o')
                correct += 1
            else:
                plt.plot(X[i][0], X[i][1], color[output] + 'x')

        print("The correctness of discriminative model:")
        print(correct / len(data))
        print("----------------------------------")

        if plot:
            plt.show()
        plt.close()

class generative_model:

    def __init__(self):
        self.theta = np.zeros(3)
        self.e = np.zeros ((3, 2))
        self.cov = np.zeros ((3, 2, 2))

    def train(self, data):
        X = data[:, 0:2]
        label = data[:, 2].astype(np.int16)
        for i in range(len(X)):
            self.theta[label[i]] += 1
            self.e[label[i]] += X[i]     
        self.e = self.e / self.theta.reshape(3, 1)

        for i in range(len(X)):
            nowx = X[i:i+1]
            nowy = label[i]
            self.cov[nowy] += (nowx - self.e[nowy]).T * (nowx - self.e[nowy])
        self.cov /= self.theta.reshape(3, 1, 1)

        self.theta /= len(X)

    def classify(self, X):
        p = [multivariate_normal(self.e[i], self.cov[i]).pdf(X) * self.theta[i] for i in range(3)]
        return np.argmax(p)
        
    def predict(self, data, plot = True):

        color = ['r', 'g', 'b']

        correct = 0
        for point in data:
            output = self.classify(point[0:2])
            if(output == point[2]):
                plt.plot(point[0], point[1], color[output] + 'o')
                correct += 1
            else:
                plt.plot(point[0], point[1], color[output] + 'x')
        print("The correctness of generative model:")
        print(correct / len(data))
        print("----------------------------------")
        if plot:
            plt.show()
        plt.close()


generate_data(plot = True)

data = load_data()

modelA = discriminative_model()
modelA.train(data, 1000, 20, 0.1)
modelA.predict(data, plot = True)

modelB = generative_model()
modelB.train(data)
modelB.predict(data, plot = True)
