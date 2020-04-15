#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def createDataset(means, cov, scale=(1000, 1000, 1000), plot=False):
    data = np.ndarray((0, 3))
    for i in range(3):
        data = np.row_stack((data, np.column_stack((
            np.random.multivariate_normal(means[i], cov, scale[i]), [i] * scale[i]))))

    if plot:
        fig = plt.figure(figsize=(12, 8), dpi=100)
        plt.scatter(data[:scale[0], 0], data[:scale[0], 1], c='r', s=10, label='class A')
        plt.scatter(data[scale[0]: -scale[2], 0], data[scale[0]: -scale[2], 1], c='g', s=10, label='class B')
        plt.scatter(data[-scale[2]:, 0], data[-scale[2]:, 1], c='b', s=10, label='class C')
        plt.legend()
        plt.savefig('data.png')

    np.random.shuffle(data)
    data = ['{:.4f} {:.4f} {:.0f}'.format(*d) for d in data]

    with open('data.data', 'w') as dat:
        dat.write('\n'.join(data))


def loadDataset(fname='data.data'):
    with open(fname, 'r') as dat:
        data = np.array([np.array(row.split()) for row in dat], dtype=np.float32)
    return data


def train_test_split(data, proportion=0.8):
    dat = data.copy()
    np.random.shuffle(dat)
    train_size = int(proportion * dat.shape[0])
    train_data, test_data = data[:train_size, :-1], data[train_size:, :-1]
    train_labels, test_labels = data[:train_size, -1], data[train_size:, -1]
    return train_data, train_labels, test_data, test_labels


class LinearGenerativeModel:

    def __init__(self):
        self.__name__ = 'Linear Generative Model'

    @property
    def name(self):
        return self.__name__

    def train(self, data, labels):
        print('Start training', self.__name__)
        self.class_num = len(set(labels))
        index = [np.where(labels==i)[0] for i in range(self.class_num)]
        self.phi = [len(index[i])/labels.shape[0] for i in range(self.class_num)]
        self.mu = [np.sum(data[index[i]], axis=0)/len(index[i]) for i in range(self.class_num)]
        self.sigma = np.sum([np.matmul(self.bias(data[i], int(labels[i])), self.bias(data[i], int(labels[i])).T)
                             for i in range(data.shape[0])], axis=0) / data.shape[0]
        print('Train completed.')

    def predict(self, data):

        def p(x):
            prob = [np.matmul(np.matmul(self.bias(x, i).T, np.linalg.inv(self.sigma)),
                              self.bias(x, i)) for i in range(self.class_num)]
            return min(range(self.class_num), key=lambda i: prob[i])
        return np.array([p(x) for x in data])

    def bias(self, x, i):
        """bias between x & mu[i]

        implemented for convenience"""
        return (x - self.mu[i]).reshape(x.shape[0], 1)


class LinearDiscriminativeModel:

    def __init__(self):
        self.__name__ = 'Linear Discriminative Model'
        self.history = {'loss': list()}

    @property
    def name(self):
        return self.__name__

    def train(self, data, labels, learning_rate=1e-3, batch_size=256, epochs=1000, verbose=True, plot=False):
        print('Start training', self.__name__)
        self.class_num = len(set(labels))
        dat = np.column_stack((data, [1] * data.shape[0]))
        self.w = np.random.rand(dat.shape[1], self.class_num)
        lr = learning_rate

        for i in range(epochs):
            index = np.random.choice(range(dat.shape[0]), batch_size)
            x = dat[index]
            y = labels[index]
            self.w -= lr * np.matmul(x.T, self.softmax(np.matmul(x, self.w)) - self.one_hot(y))
            self.history['loss'].append(self.loss(x, y))

            if verbose:
                if (i + 1) % 100 == 0:
                    print('Epoch', i+1, ',Loss:', self.history['loss'][-1])

            # Learning rate reduction
            if i >= 100 and sorted(self.history['loss'][-5:]) == self.history['loss'][-5:] and lr > 1e-8:
                lr /= 2

        if plot:
            fig = plt.figure(figsize=(12, 8), dpi=100)
            plt.plot(range(epochs), self.history['loss'], label='cross entropy loss')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss variation through linear discrimination model training')
            plt.legend()
            plt.savefig(fname='Loss.png')

        print('Train completed.')

    def predict(self, data):
        dat = np.column_stack((data, [1] * data.shape[0]))
        prob = self.softmax(np.matmul(dat, self.w))
        return np.array([max(range(self.class_num), key=lambda x: p[x]) for p in prob])

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape((len(x), 1))

    def one_hot(self, labels):
        sparse_labels = np.zeros((len(labels), self.class_num))
        for i, label in enumerate(labels):
            sparse_labels[i, int(label)] = 1
        return sparse_labels

    def loss(self, data, labels):
        return -np.trace(np.matmul(self.one_hot(labels).T,
                                   np.log(self.softmax(np.matmul(data, self.w))))) / data.shape[0]


def runModel(model, *dats):
    train_data, train_labels, test_data, test_labels = dats

    linear_model = model()
    linear_model.train(train_data, train_labels)
    predict = linear_model.predict(test_data)

    print(linear_model.name + ' accuracy:', np.average(predict == test_labels), '\n')


if __name__ == "__main__":
    createDataset(means=[[0, -1], [1, 1], [-1, 1]],
                  cov=np.eye(2) * 0.2,
                  plot=True)
    runModel(LinearGenerativeModel, *train_test_split(loadDataset()))
    runModel(LinearDiscriminativeModel, *train_test_split(loadDataset()))
