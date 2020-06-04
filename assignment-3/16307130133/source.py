import numpy as np
import matplotlib.pyplot as plt
import random as rd
import scipy.stats as st


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
    draw_pic(res, "data_with_label")

    return res


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
    plt.clf()


class GMM:
    def __init__(self, raw_data, test_size=0.1, n=3, reg_cov=1e-05):
        """
        :param raw_data: array like [[x, y, label], [x, y, label] ..]
        :param test_size:
        :param n:
        """
        self.raw_data = raw_data
        self.ref_train_data, self.ref_test_data = self.split_train_test(test_size)
        self.train_data = np.array(np.array(self.ref_train_data)[:, 0:2], dtype=np.float64)
        self.test_data = np.array(np.array(self.ref_test_data)[:, 0:2], dtype=np.float64)
        self.n_models = n
        self.n_dims = len(raw_data[0]) - 1
        self.means = np.random.randint(self.train_data.min(), self.train_data.max(),
                                       size=(self.n_models, self.n_dims))

        self.covs = np.zeros((self.n_models, self.n_dims, self.n_dims))
        # print(self.covs)
        for i in range(self.n_models):
            np.fill_diagonal(self.covs[i], 1)
        self.weights = np.ones(self.n_models) / self.n_models
        self.reg_cov = reg_cov * np.identity(self.n_dims)
        # print(self.means)
        # print(self.covs)
        # print(self.weights)

    def train(self, max_iter=200):
        n_samples = len(self.train_data)
        prob_matrix = np.zeros((n_samples, self.n_models))
        for i in range(max_iter):
            for k in range(self.n_models):
                # E step
                # print(len(self.means), len(self.covs))
                self.covs += self.reg_cov
                gaussian = st.multivariate_normal(self.means[k], cov=self.covs[k])
                prob_matrix[:, k] = self.weights[k] * gaussian.pdf(self.train_data)
            total = prob_matrix.sum(axis=1)  # 以竖轴为基准，同行相加
            total[total == 0] = self.n_models
            prob_matrix /= total.reshape(-1, 1)

            # M step
            for k in range(self.n_models):
                n_k = np.sum(prob_matrix[:, k], axis=0)
                self.means[k] = (1 / n_k) * np.sum(self.train_data * prob_matrix[:, k].reshape(-1, 1), axis=0)
                self.covs[k] = (1 / n_k) * np.dot(
                    (prob_matrix[:, k].reshape(-1, 1) * (self.train_data - self.means[k])).T,
                    (self.train_data - self.means[k])) + self.reg_cov
                self.weights[k] = n_k / n_samples
            if i % 100 == 0:
                print("Iter:", i)
                self.score(self.predict())

    def predict(self):
        prob_matrix = np.zeros((self.test_data.shape[0], self.n_models))
        for k in range(self.n_models):
            gaussian = st.multivariate_normal(self.means[k], self.covs[k])
            prob_matrix[:, k] = self.weights[k] * gaussian.pdf(self.test_data)
        total = prob_matrix.sum(axis=1)
        total[total == 0] = self.n_models
        prob_matrix /= total.reshape(-1, 1)
        return np.argmax(prob_matrix, axis=1)

    def score(self, predict):
        max_score = 0
        max_cnt = 0
        right_case = []
        cases = [['A', 'B', 'C'], ['A', 'C', 'B'], ['B', 'A', 'C'], ['B', 'C', 'A'], ['C', 'B', 'A'], ['C', 'A', 'B']]
        for case in cases:
            cnt = 0
            for index in range(len(self.ref_test_data)):
                if self.ref_test_data[index][2] == case[predict[index]]:
                    cnt += 1
            score = cnt / len(self.ref_test_data)
            if score > max_score:
                max_score = score
                max_cnt = cnt
                right_case = case
        print("Accuracy: %d/%d, %f" % (max_cnt, len(self.ref_test_data), max_score))
        res = []
        for index in range(len(self.test_data)):
            one = []
            one.extend(self.test_data[index])
            one.append(right_case[predict[index]])
            res.append(one)
        draw_pic(res, "test_predict")

    def split_train_test(self, test_size):
        train = []
        test = []
        rd.shuffle(self.raw_data)
        cnt = np.ceil(len(self.raw_data) * test_size)
        for d in self.raw_data:
            if cnt > 0:
                test.append(d)
                cnt -= 1
            else:
                train.append(d)
        return train, test


if __name__ == '__main__':
    mean1 = [1, 20]
    mean2 = [10, 10]
    mean3 = [20, 5]
    d = 5
    cov1 = [[d, 0], [0, d]]
    cov2 = [[d, 0], [0, d]]
    cov3 = [[d, 0], [0, d]]
    datas = generate_gaussian_distribution([mean1, mean2, mean3], [cov1, cov2, cov3], ['A', 'B', 'C'], [100, 100, 100])

    model = GMM(datas, test_size=0.2, n=3)
    model.train(200)
    res_predict = model.predict()

    # print(res_predict)
    model.score(res_predict)
