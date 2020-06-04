import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import joypy
from sklearn import datasets

def multivariate_normal(mean, cov, sample_num, expand_size=None):
    """
    generate sample_num data in single class with rejection sampling
    data range: (mean - expand_size , mean + expand_size)
    Note: if expand_size not specified, expand_size would be coded as Confidence Interval to hold 99% data
    :param mean: (variant_num)
    :param cov: (variant_num, variant_num)
    :param sample_num: (1, )
    :param expand_size: (1, )
    :return: (sample_num, variant_num)
    """
    if expand_size is None:
        expand_size = np.array(cov).diagonal() * 3
    sample = []
    if mean.shape[0] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
        print("Dimension error!")
        return None
    while len(sample) != sample_num:
        u = (np.random.rand(mean.shape[0]) * 2 * expand_size - expand_size) + mean
        exp_part = -0.5 * np.mat(u - mean) * np.mat(cov) * np.mat(u - mean).T
        generator = np.random.rand(1)
        if np.exp(exp_part) >= generator:
            sample.append(u)
    return np.stack(sample, axis=0)


def multivariate_normal_boxmuller(mean, cov, sample_num):
    """
    generate sample_num data in single class with box-muller sampling
    :param mean: (variant_num)
    :param cov: (variant_num)
    :param sample_num: (1, )
    :param expand_size: (1, )
    :return: (sample_num, variant_num)
    """
    sample = []
    while len(sample) != sample_num:
        # norm_part = 1 / ((2 * np.pi) ** ((mean.shape[0] - 1) / 2) * (1/np.linalg.det(cov)) ** 1 / 2)
        u, v = np.random.rand(mean.shape[0]), np.random.rand(mean.shape[0])
        u_log = -np.log(u)
        v_the = 2 * np.pi * v
        u_sqr = np.sqrt(u_log)
        x = u_sqr * np.cos(v_the)
        data = x * cov + mean
        sample.append(data)
    return np.stack(sample, axis=0)


def normal_distribution_generate(mean, cov, class_num, sample_num):
    """
    create datasets with parameters
    :param mean: (class_num, variant_num)
    :param cov: (class_num, variant_num, variant_num)
    :param class_num: (1, )
    :param sample_num: (class_num, )
    :return: sample: (sample_num, variant_num + label_num)
    """
    sample_list = []
    for i in range(class_num):
        data_raw = multivariate_normal(mean[i, :], cov[i, :, :], sample_num[i]).T
        # data_raw = multivariate_normal_boxmuller(mean[i, :], cov[i, :], sample_num[i]).T
        # labels = np.ones_like(data_raw[0, :]) * i
        # data = np.concatenate([data_raw, labels[None, :]], axis=0)
        # plt.plot(data[0, :], data[1, :], 'x')
        sample_list.append(data_raw)
    sample = np.concatenate(sample_list, axis=1)
    sample = np.transpose(sample, (1, 0))
    plt.plot(sample[:, 0], sample[:, 1], '.')
    plt.title("Rejection Sampling")
    plt.show()
    return sample


def normal_distribution_generate_boxmuller(mean, cov, class_num, sample_num):
    """
    create datasets with parameters
    :param mean: (class_num, variant_num)
    :param cov: (class_num, variant_num, variant_num)
    :param class_num: (1, )
    :param sample_num: (class_num, )
    :return: sample: (sample_num, variant_num + label_num)
    """
    sample_list = []
    for i in range(class_num):
        # data_raw = multivariate_normal(mean[i, :], cov[i, :, :], sample_num[i]).T
        data_raw = multivariate_normal_boxmuller(mean[i, :], cov[i, :], sample_num[i]).T
        # labels = np.ones_like(data_raw[0, :]) * i
        # data = np.concatenate([data_raw, labels[None, :]], axis=0)
        # plt.plot(data[0, :], data[1, :], 'x')
        sample_list.append(data_raw)
    sample = np.concatenate(sample_list, axis=1)
    sample = np.transpose(sample, (1, 0))
    plt.plot(sample[:, 0], sample[:, 1], '.')
    plt.title("Box-Muller Sampling")
    plt.show()
    return sample


def save_dataset(sample, file_name='data.data'):
    """
    Save the data and create dataset
    :param sample: (sample_num, variant_num + label)
    :param file_name: string
    :return: file with file_name
    """
    np.random.shuffle(sample)
    np.savetxt(file_name, sample, fmt='%f', delimiter=',')


def load_dataset(file_name):
    """
    laod dataset
    :param file_name:name of file
    :return: sample: (sample_num, variant_num + label)
    """
    sample = np.loadtxt(file_name, delimiter=',')
    return sample


class GaussianMixtureModel:
    def __init__(self, k=3, maxiter=50, file_name='data.data'):
        # data
        self.file_name = file_name                          # string
        self.data = load_dataset(self.file_name)            # (sample_num, variant_num + 1(label))
        self.x = self.data[:, :]                            # (sample_num, variant_num)
        self.variant_num = self.data.shape[1]               # variant_num
        self.n = self.data.shape[0]                         # sample_num

        # learn
        self.k = k                                          # int - class_num
        self.gamma = np.zeros((self.n, self.k))             # (sample_num, k)
        self.count = maxiter                                # int - iteration times

        # parameter
        self.mean = np.array([self.x[i, :] for i in range(k)])                           # (k, variant_num)
        self.cov = np.repeat(np.identity(self.variant_num)[None, :, :], self.k, axis=0)  # (k, variant_num, variant_num)
        self.pi = np.ones(self.k) / self.k                                               # (k, )

    def train(self):
        """
        train the model for self.count times
        """
        expect_record = []
        for __ in range(self.count):
            self.gamma, _ = self.y_expect_cal(self.x, self.pi, self.mean, self.cov)
            self.pi, self.mean, self.cov = self.para_cal(self.x, self.gamma)
            _, expect = self.y_expect_cal(self.x, self.pi, self.mean, self.cov)
            expect_record.append(expect)
        plt.plot(expect_record, linestyle='dashed')
        plt.title('GMM Expectation: k = %d' % self.k)
        plt.xlabel('iteration')
        plt.ylabel('expectation')
        plt.show()

        self.gamma, _ = self.y_expect_cal(self.x, self.pi, self.mean, self.cov)
        label = np.argmax(self.gamma, axis=1)
        plt.scatter(self.x[:, 0], self.x[:, 1], c=label, marker='.')
        plt.scatter(self.mean[:, 0], self.mean[:, 1], marker='x')
        plt.title("Visualizing GMM Clustering")
        plt.show()

    def y_expect_cal(self, x, pi, mean, cov):
        """
        calculate gamma and loss with the fixed pi, mean and cov
        :param x: (sample_num, variant_num)
        :param pi: (k, )
        :param mean: (k, variant_num)
        :param cov: (k, variant_num, variant_num)
        :return:(sample_num, k) & float
        """
        n_list = []
        expect = 0
        for n in range(x.shape[0]):
            k_list = []
            for k in range(mean.shape[0]):
                norm_part = 1 / ((2 * np.pi) ** ((x.shape[1]) / 2) * ((np.linalg.det(cov[k, :, :])) ** 0.5))
                vec = (x[n, :] - mean[k, :])
                exp_part = -0.5 * np.dot(np.dot(vec, np.linalg.pinv(cov[k, :, :])), vec.T)
                k_list.append(norm_part * np.exp(exp_part) * pi[k])
            k_list = np.stack(k_list, axis=0)
            k_sum = k_list.sum()
            n_list.append(k_list/k_sum)
            expect += np.log(k_sum)
        return np.stack(n_list, axis=0), expect

    def para_cal(self, x, gamma):
        """
        calculate pi, means, sigma with the fixed gamma
        :param x: (sample_num, variant_num)
        :param gamma: (sample_num, k)
        :return: (k, ), (k, variant_num) and (k, variant_num, variant_num)
        """
        pi = np.sum(gamma, axis=0)
        u_list, sigma_list = [], []
        for k in range(pi.shape[0]):
            uk = 0
            sigmak = np.zeros((x.shape[1], x.shape[1]))
            for n in range(x.shape[0]):
                uk += gamma[n, k] * x[n, :]
            uk = uk / pi[k]
            for n in range(x.shape[0]):
                sigmak += gamma[n, k] * np.dot((x[n, :] - uk)[:, None], (x[n, :].squeeze() - uk)[None, :])
            sigmak = sigmak / pi[k]
            u_list.append(uk)
            sigma_list.append(sigmak)
        return pi / gamma.shape[0], np.stack(u_list, axis=0), np.stack(sigma_list, axis=0)


# K-Means Model for comparision with GMM
class KMeansModel:
    def __init__(self, k=3, maxiter=200, file_name='data.data', center=None):
        # data
        self.file_name = file_name
        self.data = load_dataset(self.file_name)
        self.x = self.data[:, :]                            # (sample_num, variant_num)
        self.k = k                                          # k
        self.variant_num = self.x.shape[1]                  # variant_num
        self.n = self.x.shape[0]                            # sample_num
        if center is None:                                  # (k, variant_num)
            self.center = np.stack([self.x[i, :] for i in range(self.k)], axis=0)
        else:
            self.center = np.array(center)
        self.label = np.zeros(self.n)                       # (sample_num, )
        self.iter = maxiter                                 # int iteration_times

    def train(self):
        for _ in range(self.iter):
            self.label = self.label_cal(self.x, self.center)
            self.center = self.center_cal(self.x, self.label, self.k)
        self.label = self.label_cal(self.x, self.center)

        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.label, marker='.')
        plt.scatter(self.center[:, 0], self.center[:, 1], marker='x')
        plt.title("Visualizing K-Means Clustering")
        plt.show()

    def label_cal(self, x, center):
        """
        calculate the labels with distance to fixed centers
        :param x: (sample_num, variant_num)
        :param center: (k, variant_num)
        :return: (sample_num, )
        """
        label = []
        for n in range(x.shape[0]):
            k_list = []
            for k in range(center.shape[0]):
                k_list.append(((x[n, :] - center[k, :]) ** 2).sum())
            k_np = np.stack(k_list, axis=0)
            label.append(k_np.argmin())
        return np.stack(label, axis=0)

    def center_cal(self, x, label, k_num):
        """
        calculate the centers with fixed labels and k
        :param x: (sample_num, variant_num)
        :param label: (sample_num, )
        :param k_num: int k
        :return: (k, variant_num)
        """
        data = np.concatenate((x, label[:, None]), axis=1)
        center = []
        for k in range(k_num):
            dots = data[data[:, -1] == k]
            center.append(dots[:, :-1].mean(axis=0))
        return np.stack(center, axis=0)