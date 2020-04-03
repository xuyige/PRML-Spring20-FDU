"""
Source
======

Provides
    1. Linear discriminative and generative model for classification.
    2. Functions used for create, load and split dataset.

Examples
--------
>>> create_dataset()
>>> samples, labels = load_dataset()
>>> set_of_samples, set_of_labels = split_dataset(samples, labels, [0.8])
>>> training_samples, testing_samples = set_of_samples
>>> training_labels, testing_labels = set_of_labels
>>> model1 = LinearDiscriminativeModel()
>>> model1.train(training_samples, training_labels, max_epochs=20)
Epoch	Accuracy
1		0.9590163934426229 #random
2		0.9639344262295082
3		0.9655737704918033
4		0.9672131147540983
5		0.9688524590163935
6		0.9688524590163935
7		0.9672131147540983
8		0.9704918032786886
9		0.9721311475409836
10		0.9721311475409836
11		0.9721311475409836
12		0.9737704918032787
13		0.9737704918032787
14		0.9737704918032787
15		0.9737704918032787
16		0.9737704918032787
17		0.9737704918032787
18		0.9737704918032787
19		0.9737704918032787
20		0.9737704918032787
>>> labels1 = model1.classify(testing_samples)
>>> acc = sum(testing_labels == labels1) / testing_labels.size
>>> print(model1.w)
[[ 2.62622022 -1.31718143 -1.30903878]  #random
 [ 0.72619941 -3.81306407  3.08686466]]
>>> print(acc[0, 0])
0.9794871794871794  #random
>>> model2 = LinearGenerativeModel()
>>> model2.train(training_samples, training_labels)
>>> labels2 = model2.classify(testing_samples)
>>> acc = sum(testing_labels == labels2) / testing_labels.size
>>> print(model2.w)
[[ 9.18231285 -0.0287646   0.49775292]  #random
 [ 0.27302244 -9.82742259  9.82056657]]
>>> print(acc[0, 0])
0.9794871794871794 #random

"""
import numpy as np


def generate_gaussian_distributed_samples(mus, sigma, ps, n):
    """
    Generate gaussian distributed samples from specified expectation and covariance matrix.

    Parameters
    ----------
    mus : array_like
        Expectations of gaussian distributions.
    sigma : array_like
       Covariance matrix of gaussian distributions.
    ps:
        Sampling probability of each class.
    n : int
        number of samples

    Returns
    -------
    samples :
        samples compiled to gaussian distribution
    labels :
        Sample labels.

    """

    mus, sigma, ps = np.asarray(mus), np.asarray(sigma), np.asarray(ps)
    ns = np.around(ps * n).astype(int)
    samples, labels = np.zeros((n, mus.shape[1])), np.zeros(n)
    for i in range(ns.size):
        mu, ni = mus[i, :], ns[i]
        beg, end = sum(ns[:i]), sum(ns[:i + 1])
        samples[beg:end, :] = np.random.multivariate_normal(mu, sigma, (ni,))
        labels[beg:end] = [i] * (end - beg)
    index = np.arange(n)
    np.random.shuffle(index)
    samples, labels = samples[index, :], labels[index]
    labels.shape = (n, 1)
    return samples, labels


def create_dataset(mus=None, ps=None, sigma=None, n=1000):
    """
    Create data set for classification and output to txt file
    """
    if mus is None:
        mus = [[1, 0], [0, -1], [0, 1]]
    if ps is None:
        ps = [.3, .6, .1]
    if sigma is None:
        sigma = [[.1, 0], [0, .1]]

    samples, labels = generate_gaussian_distributed_samples(mus, sigma, ps, n)
    dataset = ''
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            dataset = dataset + '{:.4f} '.format(samples[i, j])
        dataset = dataset + str(int(labels[i, 0])) + '\n'
    f = open('*.data', 'w')
    f.write(dataset)
    f.close()


def load_dataset(filename=None):
    """
    load samples and labels from txt file
    """
    if filename is None:
        filename = '*.data'
    samples = np.loadtxt(filename)
    n = samples.shape[0]
    samples, labels = samples[:, 0:-1], samples[:, -1].astype(int).reshape((n, 1))
    return samples, labels


def split_dataset(samples, labels, proportions):
    """
    Split dataset in proportion.

    Parameters
    ----------
    samples : array_like
        Samples in rows.
    labels : array_like
        Sample labels from 0 to n-1.
    proportions : list of float
        Proportions of each part of dataset from 0 ot 1.
        If the sum of proportions is no 1, a new part will be added.

    Return
    ------
    xs : list of samples
    ys : list of labels

    Example
    -------
    >>> samples, labels = split_dataset(samples, labels, [0.8])

    """
    xs, ys = [], []
    n = samples.shape[0]
    idx = np.arange(n).astype(int)
    np.random.shuffle(idx)
    if not np.sum(proportions) == 1:
        proportions.extend([1 - np.sum(proportions)])
    nums = np.round(n * np.asarray(proportions)).astype(int)
    beg, end = 0, 0
    for i in range(nums.size):
        if i == nums.size - 1:
            end = n
        else:
            end = beg + nums[i]
        xs.extend([samples[idx[beg: end], :]])
        ys.extend([labels[idx[beg: end], :]])
        beg = end
    return xs, ys


class LinearDiscriminativeModel:
    """
    Linear discriminative model for classification with softmax regression and argmax classification strategy
    """

    def __init__(self):
        self.n_classes = 0
        self.w = np.zeros((0,))

    def train(self, samples, labels, learning_rate=0.9, max_epochs=50, mini_batch_size=128):
        """
        Train the model using samples and labels

        Parameters
        ----------
        samples : array_like
            Training samples in rows.
        labels : array_like
            Sample labels from 0 to n-1.
        learning_rate : float, optional
            Learning rate of softmax regression.
        max_epochs : int, optional
            Max number of training epochs.
        mini_batch_size: int, optional
            Mini batch size of training.

        Examples
        --------
        >>> model = LinearDiscriminativeModel()
        >>> model.train(samples, labels, max_epochs=100)

        """
        xs = np.asmatrix(samples)
        n = xs.shape[0]
        self.n_classes = int(labels.max() + 1)
        ys = np.asmatrix(np.zeros((self.n_classes, n)))
        alpha, iter_times = learning_rate, int(np.ceil(n / mini_batch_size))
        for i in range(self.n_classes):
            idx = (labels == i).reshape((labels.size,))
            ys[i, idx] = 1
        self.w = np.asmatrix(np.zeros((xs.shape[1], self.n_classes)))

        print('Epoch\tAccuracy')
        for epoch in range(max_epochs):
            for i in range(iter_times):
                beg = mini_batch_size * i
                if i == iter_times - 1:
                    end = n
                else:
                    end = mini_batch_size * (i + 1) + 1
                x, y = xs[beg: end, :].T, ys[:, beg: end].T
                wx = self.w.T.dot(x).T
                y1 = softmax(wx)
                delta = np.asmatrix(np.zeros(self.w.shape))
                for j in range(x.shape[1]):
                    delta = delta + x[:, j].dot(y[j, :] - y1[j, :])
                delta = delta / x.shape[1]
                self.w = self.w + alpha * delta
            labels1 = self.classify(samples)
            acc = sum(labels == labels1) / labels.size
            print('{epoch}\t\t{acc}'.format_map({"epoch": epoch + 1, "acc": acc[0, 0]}))

    def classify(self, samples):
        """
        classify input samples

        Parameters
        ----------
        samples : array_like
            Samples in rows.

        Returns
        -------
        labels : list of int
            Sample labels from 0 to n-1.

        Examples
        --------
        >>> model = LinearDiscriminativeModel()
        >>> model.train(training_samples, training_labels)
        >>> pred = model.classify(testing_samples)
        >>> acc = sum(pred == testing_labels) / testing_labels.size

        """
        xs = np.asmatrix(samples)
        wx = self.w.T.dot(xs.T).T
        ys = softmax(wx)
        labels = np.argmax(ys, axis=1)
        return labels


class LinearGenerativeModel:
    """
    Linear generative model for classification using gaussian distributions
    """
    def __init__(self):
        self.n_classes = 2
        self.w = np.zeros((0,))
        self.b = np.zeros((0,))
        self.sigma = np.zeros((0,))

    def train(self, samples, labels):
        """
        Train the model using samples and labels

        Parameters
        ----------
        samples : array_like
            Training samples in rows.
        labels : array_like
            Sample labels from 0 to n-1.

        Examples
        --------
        >>> model = LinearGenerativeModel()
        >>> model.train(samples, labels)

        """
        xs = np.asmatrix(samples)
        self.n_classes = int(labels.max() + 1)
        mus = np.asmatrix(np.zeros((xs.shape[1], self.n_classes)))
        self.w = np.asmatrix(np.zeros((xs.shape[1], self.n_classes)))
        self.b = np.asmatrix(np.zeros((self.n_classes, 1)))
        self.sigma = np.asmatrix(np.zeros((xs.shape[1], xs.shape[1])))
        ps = np.asmatrix([sum(labels == i) / xs.shape[0] for i in range(self.n_classes)])
        for i in range(self.n_classes):
            idx = (labels == i).reshape((labels.size,))
            mus[:, i] = np.mean(xs[idx, :], axis=0).T
            self.sigma = self.sigma + ps[i, 0] * np.cov(samples[idx, :].T)
        sigma_inv = self.sigma.I
        for i in range(self.n_classes):
            self.w[:, i] = sigma_inv.dot(mus[:, i])
            self.b[i, 0] = -1 / 2 * mus[:, i].T.dot(self.w[:, i]) + np.log(ps[i, 0])

    def classify(self, samples):
        """
        classify input samples

        Parameters
        ----------
        samples : array_like
            Samples in rows.

        Returns
        ----------
        labels : list of int
            Sample labels from 0 to n-1.

        Examples
        ----------
        >>> model = LinearGenerativeModel()
        >>> model.train(training_samples, training_labels)
        >>> pred = model.classify(testing_samples)
        >>> acc = sum(pred == testing_labels) / testing_labels.size

        """
        a = (self.w.T.dot(samples.T) + self.b).T
        ys = softmax(a)
        labels = np.argmax(ys, axis=1)
        return labels


def softmax(x):
    x = np.asmatrix(x)
    x = x - np.max(x, axis=1)
    return np.exp(x) / np.sum(np.exp(x), axis=1)


def main():
    pass


if __name__ == '__main__':
    main()
