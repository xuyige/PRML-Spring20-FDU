import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import random as rd


def generate_gaussian_distribution(means, covs, labels, n=[100, 100, 100]):
    """
    :param means: array like [[m1, m2], [..], [..]], three 2-d means of gaussian distribution.
    :param covs: array like [[2x2], [2x2], [2x2]], three 2x2 covs of gaussian distribution.
    :param labels: array like ['A', 'B', 'C'], three label names.
    :param n: [,,]amount of points in each gaussian distribution.
    :return: [[x1, x2, label], [..], ..]
    """
    mean1 = means[0]
    mean2 = means[1]
    mean3 = means[2]

    cov1 = covs[0]
    cov2 = covs[1]
    cov3 = covs[2]

    data1 = np.random.multivariate_normal(mean1, cov1, n[0])
    data2 = np.random.multivariate_normal(mean2, cov2, n[1])
    data3 = np.random.multivariate_normal(mean3, cov3, n[2])

    res = []
    for data in data1:
        res.append([data[0], data[1], labels[0]])
    for data in data2:
        res.append([data[0], data[1], labels[1]])
    for data in data3:
        res.append([data[0], data[1], labels[2]])
    draw_pic(res, "Gaussian")
    return res


class GenerativeModel:
    def __init__(self, data):
        self.data = data  # [[x1, x2, label], ...]
        pass

    def train(self):
        mus, b, pi = self.compute_param()
        b_share = np.zeros((2, 2))
        for i in range(0, 3):
            b_share += b[i] * pi[i]
        labels = ['A', 'B', 'C']
        right = 0
        predict = []
        for i in range(len(self.data)):
            x = [self.data[i][0], self.data[i][1]]
            predict.append(x)
            p_x = np.zeros(3)
            sum = 0
            for index in [0, 1, 2]:
                # we can substitute b[index] with b_share
                p_x[index] = st.multivariate_normal(mus[index], b_share).pdf(x) * pi[index]
                sum += p_x[index]
            p_x /= sum
            max_p = 0
            predict_label = ''
            for index in [0, 1, 2]:
                if max_p < p_x[index]:
                    max_p = p_x[index]
                    predict_label = labels[index]
            predict[-1].append(predict_label)
            if predict_label == self.data[i][2]:
                right += 1
        print("Total:%d, predict right:%d, accuracy:%f" % (len(self.data), right, right * 1.0 / len(self.data)))
        draw_pic(predict, "GenerativeModelPredict")
        pass

    def compute_param(self):
        """
        :return: [mu1, mu2, mu3], [b1, b2, b3] Maximum Likelihood Estimate
        """
        data1 = []
        data2 = []
        data3 = []
        n1 = 0
        n2 = 0
        n3 = 0
        for d in self.data:
            if d[2] == 'A':
                data1.append([d[0], d[1]])
                n1 += 1
            elif d[2] == 'B':
                data2.append([d[0], d[1]])
                n2 += 1
            elif d[2] == 'C':
                data3.append([d[0], d[1]])
                n3 += 1
        mus = np.zeros((3, 2))
        b = np.zeros((3, 2, 2))  # share cov matrix
        pi = np.zeros(3)
        mus[0] = self.get_mu(data1, n1)
        mus[1] = self.get_mu(data2, n2)
        mus[2] = self.get_mu(data3, n3)

        b[0] = self.get_b(data1, n1, mus[0])
        b[1] = self.get_b(data2, n2, mus[1])
        b[2] = self.get_b(data3, n3, mus[2])
        pi += [n1, n2, n3]
        pi /= (n1 + n2 + n3)
        return mus, b, pi

    @staticmethod
    def get_mu(data, n):
        mu = np.zeros(2)
        for i in data:
            mu += i
        return mu / n

    @staticmethod
    def get_b(data, n, mu):
        b = np.zeros((2, 2))
        for i in data:
            b += np.array([i - mu]).T * (i - mu)
        return b / n


class DiscriminativeModel:
    def __init__(self, data):
        self.data = data
        self.train_data, self.test_data = self.split_data()
        self.w = np.zeros((3, 2))
        self.b = np.zeros((3, 1))

    def train(self, batch, epoch, learning_rate):
        predict = np.zeros((len(self.train_data), 3))
        labels = ['A', 'B', 'C']
        iter_time = int(np.ceil(len(self.train_data) / batch))
        for one_epoch in range(0, epoch):
            rd.shuffle(self.train_data)
            index = 0
            right = 0
            predict_result = []
            for it in range(0, iter_time):
                if index == len(self.train_data):
                    break
                for i in range(0, batch):
                    partial_w = np.zeros((3, 2))
                    partial_b = np.zeros((3, 1))
                    x = np.array(self.train_data[index][0:2])
                    predict_result.append(self.train_data[index][0:2])
                    label = self.train_data[index][2]
                    y_true = np.zeros(3)
                    if label == 'A':
                        y_true[0] = 1
                    elif label == 'B':
                        y_true[1] = 1
                    elif label == 'C':
                        y_true[2] = 1
                    max_p = -1
                    predict_label = ''
                    for turn in range(0, 3):
                        predict[index][turn] = self.sigmod(np.dot(self.w[turn].T, x) + self.b[turn])
                        partial_w[turn] += np.dot((y_true[turn] - predict[index][turn]), x)
                        partial_b[turn] += (y_true[turn] - predict[index][turn])
                        if predict[index][turn] > max_p:
                            max_p = predict[index][turn]
                            predict_label = labels[turn]
                    predict_result[-1].append(predict_label)
                    if predict_label == label:
                        right += 1
                    self.w += partial_w / batch * learning_rate
                    self.b += partial_b / batch * learning_rate
                    index += 1
            print("Epoch:", one_epoch)
            print("Train:\tTotal:%d, right:%d, accuracy:%f" % (
                len(self.train_data), right, right * 1.0 / len(self.train_data)))

    def test(self):
        labels = ['A', 'B', 'C']
        right = 0
        predict_result = []
        draw_pic(self.test_data, "DiscriminativeTestTrue")
        for d in self.test_data:
            x = np.array(d[0:2])
            predict_result.append(d[0:2])
            label = d[2]
            predict = np.zeros(3)
            max_p = -1
            predict_label = ''
            for index in range(0, 3):
                predict[index] = self.sigmod(np.dot(self.w[index].T, x) + self.b[index])
                if predict[index] > max_p:
                    max_p = predict[index]
                    predict_label = labels[index]
            predict_result[-1].append(predict_label)
            if predict_label == label:
                right += 1
        draw_pic(predict_result, "DiscriminativeTestPredict")
        print(
            "Test:\tTotal:%d, right:%d, accuracy:%f" % (len(self.test_data), right, right * 1.0 / len(self.test_data)))

    @staticmethod
    def sigmod(x):
        return 1.0 / (1 + np.exp(-x))

    def split_data(self, percent=0.8):
        """
        :param percent: the split percent of total data 
        :return: the percent of data used as training data, rest for testing.
        """""
        train = []
        test = []
        rd.shuffle(self.data)
        cnt = np.ceil(len(self.data) * percent)
        for d in self.data:
            if cnt > 0:
                train.append(d)
                cnt -= 1
            else:
                test.append(d)
        return train, test


def draw_pic(data, title):
    c1 = [[], []]
    c2 = [[], []]
    c3 = [[], []]
    for i in range(len(data)):
        if data[i][2] == 'A':
            c1[0].append(data[i][0])
            c1[1].append(data[i][1])
        elif data[i][2] == 'B':
            c2[0].append(data[i][0])
            c2[1].append(data[i][1])
        elif data[i][2] == 'C':
            c3[0].append(data[i][0])
            c3[1].append(data[i][1])
    plt.scatter(c1[0], c1[1], c='red')
    plt.scatter(c2[0], c2[1], c='yellow')
    plt.scatter(c3[0], c3[1], c='green')
    plt.axis()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(['A', 'B', 'C'], loc='upper left')
    plt.savefig(title)


if __name__ == '__main__':
    mean1 = [1, 0]
    mean2 = [10, 10]
    mean3 = [20, 20]
    cov1 = [[8, 0], [0, 5]]
    cov2 = [[4, 0], [0, 4]]
    cov3 = [[3, 0], [0, 5]]
    # mean1 = [1, 0]
    # mean2 = [2, 4]
    # mean3 = [3, 6]
    # cov1 = [[8, 0], [0, 10]]
    # cov2 = [[20, 0], [0, 13]]
    # cov3 = [[18, 0], [0, 20]]
    datas = generate_gaussian_distribution([mean1, mean2, mean3], [cov1, cov2, cov3], ['A', 'B', 'C'], [100, 400, 800])
    print("----------Generative Model----------")
    ob = GenerativeModel(datas)
    ob.train()
    print("----------Discriminative Model----------")
    ob2 = DiscriminativeModel(datas)
    ob2.train(20, 250, 1)
    ob2.test()
