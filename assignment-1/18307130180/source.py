"""
Source
======

Provides
    1. Linear discriminative and generative model for classification.
    2. Functions used for create, load and split dataset.

Examples
========
>>> from source import *
>>> create_dataset()
>>> samples, labels = load_dataset()
>>> set_of_samples, set_of_labels = split_dataset(samples, labels, [0.7])
>>> training_samples, testing_samples = set_of_samples
>>> training_labels, testing_labels = set_of_labels
>>> model1 = LinearDiscriminativeModel()
>>> model1.train(training_samples, training_labels, learning_rate=.9, max_epochs=20, plot_training_process=False)
Epoch	Accuracy
1		0.9714285714285714
2		0.9785714285714285
3		0.9814285714285714
4		0.9828571428571429
5		0.9828571428571429
6		0.9828571428571429
7		0.9828571428571429
8		0.9814285714285714
9		0.9814285714285714
10		0.9842857142857143
11		0.9842857142857143
12		0.9842857142857143
13		0.9842857142857143
14		0.9842857142857143
15		0.9842857142857143
16		0.9842857142857143
17		0.9842857142857143
18		0.9842857142857143
19		0.9842857142857143
20		0.9842857142857143
>>> labels1 = model1.classify(testing_samples)
>>> acc = sum(testing_labels == labels1) / testing_labels.size
>>> acc[0, 0]
0.98
>>> model1.w
matrix([[ 2.89275597, -1.44460552, -1.44815044],
        [ 0.77766718, -4.04375914,  3.26609195]])
>>> model2 = LinearGenerativeModel()
>>> model2.train(training_samples, training_labels)
>>> labels2 = model2.classify(testing_samples)
>>> acc = sum(testing_labels == labels2) / testing_labels.size
>>> acc[0, 0]
0.98
>>> model2.w
matrix([[  9.57567184,   0.06218358,   0.07589503],
        [ -0.11014837, -10.67945524,  11.09769756]])
>>> model2.b
matrix([[-5.85848281],
        [-5.9195002 ],
        [-8.16539317]])

"""
import matplotlib.pyplot as plt
import numpy as np


class LinearModel:
    def __init__(self):
        self.colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]])
        self.axes_size = np.array([[0.1, 0.4, 0.8, 0.5], [0.1, 0.1, 0.8, 0.25], [0.1, 0.1, 0.8, 0.8]])

    def plot_classification_result(self, axes, samples, labels, preds):
        """
        Plot the classification result.

        If the sample has more than two dimensions, this method will only plot the first two dimensions of sample. If
        the sample has only one dimension, this method will plot samples on the X axis of the plane. The
        misclassified samples are shown in black while others are shown in other colors. If samples are classified
        into more than 6 classes, some classes will be shown in the same cyan color.

        """
        labels, preds = np.asarray(labels), np.asarray(preds)
        plt.cla()

        if samples.ndim == 1:
            xs = np.hstack(samples, np.zeros(samples.shape))
        else:
            xs = samples
        idx = (labels != preds).reshape((labels.size,))
        axes.scatter(xs[idx, 0], xs[idx, 1], s=10, c='k')
        for i in range(labels.max() + 1):
            if i < self.colors.shape[0]:
                color = [self.colors[i, :]]
            else:
                color = [self.colors[-1, :]]
            idx = ((labels == preds) & (labels == i)).reshape((labels.size,))
            axes.scatter(xs[idx, 0], xs[idx, 1], s=10, c=color)


class LinearDiscriminativeModel(LinearModel):
    """
    Linear discriminative model for classification with softmax regression and argmax classification strategy.
    """

    def __init__(self):
        super().__init__()
        self.n_classes = 0
        self.w = np.zeros((0,))

    def train(self, samples, labels, learning_rate=0.9, max_epochs=50, mini_batch_size=128, plot_training_process=True):
        """
        Train the model using samples and labels.

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
        plot_training_process: bool, optional
            Plot the training process of not.
        Examples
        --------
        >>> model = LinearDiscriminativeModel()
        >>> model.train(samples, labels, max_epochs=100, mini_batch_size=64)

        """
        self.n_classes = int(labels.max() + 1)
        self.w = np.asmatrix(np.zeros((samples.shape[1], self.n_classes)))

        n = samples.shape[0]
        xs = np.asmatrix(samples)
        ys = np.asmatrix(np.zeros((self.n_classes, n)))

        alpha, iter_times = learning_rate, int(np.ceil(n / mini_batch_size))
        for i in range(self.n_classes):
            idx = (labels == i).reshape((labels.size,))
            ys[i, idx] = 1

        if plot_training_process:
            epochs, accuracies = [], []
            fig = plt.figure()
            plt.clf()
            plt.ion()
            axes1 = plt.axes(self.axes_size[0, :])
            axes2 = plt.axes(self.axes_size[1, :])
            axes1.set_title('Classification result and accuracy')

        print('Epoch\tAccuracy')
        for epoch in range(max_epochs):
            for i in range(iter_times):
                beg = mini_batch_size * i
                if i == iter_times - 1:
                    end = n
                else:
                    end = mini_batch_size * (i + 1) + 1
                x, y = xs[beg: end, :].T, ys[:, beg: end].T
                wx = x.T * self.w
                y1 = softmax(wx)
                delta = np.zeros(self.w.shape)
                for j in range(x.shape[1]):
                    delta = delta + x[:, j] * (y[j, :] - y1[j, :])
                delta = delta / x.shape[1]
                self.w = self.w + alpha * delta

            labels1 = self.classify(samples)
            acc = sum(labels == labels1) / labels.size

            if plot_training_process:
                epochs.extend([epoch + 1])
                accuracies.extend([acc[0, 0]])
                self.__plot_training_process(axes1, axes2, epochs, accuracies, samples, labels, labels1)
            print('{epoch}\t\t{acc}'.format_map({"epoch": epoch + 1, "acc": acc[0, 0]}))

        if plot_training_process:
            plt.ioff()
            plt.show()

    def classify(self, samples, plot_classification_result=False):
        """
        Classify input samples.

        Parameters
        ----------
        samples : array_like
            Samples in rows.
        plot_classification_result : bool, optional
            Plot the classification result or not.

        Returns
        -------
        labels : list of int
            Sample labels from 0 to n-1.

        Examples
        --------
        >>> model = LinearDiscriminativeModel()
        >>> model.train(training_samples, training_labels)
        >>> pred = model.classify(testing_samples)
        >>> [[acc]] = sum(pred == testing_labels) / testing_labels.size

        """
        xs = np.asmatrix(samples)
        wx = xs * self.w
        ys = softmax(wx)
        labels = np.argmax(ys, axis=1)
        if plot_classification_result:
            plt.figure()
            axes = plt.axes(self.axes_size[2, :])
            self.__plot_classification_result(axes, samples, labels, labels)
            axes.set_title('Classification result of linear discriminative model')
            plt.show()
        return labels

    def __plot_training_process(self, axes1, axes2, epochs, accuracies, samples, labels, preds):
        """
        Plot the training precess.

        The figure is divided into two parts, the upper and the lower. The upper part contains the classification
        results of samples. The lower part contains the classification accuracy and training epochs.

        """
        self.__plot_classification_result(axes1, samples, labels, preds)
        axes2.plot(epochs, accuracies)
        plt.pause(0.001)

    def __plot_classification_result(self, axes, samples, labels, preds):
        """
        Plot the classification result.
        """
        super().plot_classification_result(axes, samples, labels, preds)


class LinearGenerativeModel(LinearModel):
    """
    Linear generative model for classification using gaussian distributions.
    """

    def __init__(self):
        super().__init__()
        self.n_classes = 2
        self.w = np.zeros((0,))
        self.b = np.zeros((0,))
        self.sigma = np.zeros((0,))

    def train(self, samples, labels):
        """
        Train the model using samples and labels.

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
        N = samples.shape[1]
        self.n_classes = int(labels.max() + 1)
        self.w = np.asmatrix(np.zeros((N, self.n_classes)))
        self.b = np.asmatrix(np.zeros((self.n_classes, 1)))
        self.sigma = np.asmatrix(np.zeros((N, N)))

        xs = np.asmatrix(samples)
        mus = np.asmatrix(np.zeros((xs.shape[1], self.n_classes)))
        ps = np.asmatrix([sum(labels == i) / xs.shape[0] for i in range(self.n_classes)])

        for i in range(self.n_classes):
            idx = (labels == i).reshape((labels.size,))
            mus[:, i] = np.mean(xs[idx, :], axis=0).T
            self.sigma = self.sigma + ps[i, 0] * np.cov(samples[idx, :].T)
        sigma_inv = self.sigma.I
        for i in range(self.n_classes):
            self.w[:, i] = sigma_inv.dot(mus[:, i])
            self.b[i, 0] = -1 / 2 * mus[:, i].T * self.w[:, i] + np.log(ps[i, 0])

    def classify(self, samples, plot_classification_result=False):
        """
        classify input samples.

        Parameters
        ----------
        samples : array_like
            Samples in rows.
        plot_classification_result : bool, optional
            Plot the classification result or not.

        Returns
        ----------
        labels : list of int
            Sample labels from 0 to n-1.

        Examples
        ----------
        >>> model = LinearGenerativeModel()
        >>> model.train(training_samples, training_labels)
        >>> preds = model.classify(testing_samples)
        >>> acc = sum(preds == testing_labels) / testing_labels.size

        """
        a = samples * self.w + self.b.T
        ys = softmax(a)
        labels = np.argmax(ys, axis=1)
        if plot_classification_result:
            plt.figure()
            axes = plt.axes(self.axes_size[2, :])
            self.__plot_classification_result(axes, samples, labels, labels)
            axes.set_title('Classification result of linear generative model')
            plt.show()
        return labels

    def __plot_classification_result(self, axes, samples, labels, preds):
        """
        PLot the classification result.
        """
        super().plot_classification_result(axes, samples, labels, preds)


def generate_gaussian_distributed_samples(mus, sigma, ps, n):
    """
    Generate gaussian distributed samples from specified mathematic expectations and covariance matrix.

    Parameters
    ----------
    mus : array_like
        Mathematic expectations of Gaussian distributions.
    sigma : array_like
       Covariance matrix of Gaussian distributions.
    ps:
        Sampling probability of each class.
    n : int
        Number of samples.

    Returns
    -------
    samples :
        Samples compiled to Gaussian distributions.
    labels :
        Sample labels.

    """
    mus, sigma, ps = np.asarray(mus), np.asarray(sigma), np.asarray(ps)
    ns = np.around(ps * n).astype(int)
    samples, labels = np.zeros((n, mus.shape[1])), np.zeros(n)

    for i in range(ns.size):
        mui, ni = mus[i, :], ns[i]
        beg, end = sum(ns[:i]), sum(ns[:i + 1])
        samples[beg:end, :] = np.random.multivariate_normal(mui, sigma, (ni,))
        labels[beg:end] = [i] * (end - beg)

    index = list(range(n))
    np.random.shuffle(index)
    samples, labels = samples[index, :], labels[index].reshape((n, 1))
    return samples, labels


def create_dataset(mus=None, ps=None, sigma=None, n=1000):
    """
    Create data set for classification and output to txt file.
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
    Load samples and labels from file.
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

    Returns
    -------
    xs :
        List of samples.
    ys :
        List of labels.

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


def softmax(x):
    """
    Non overflow and underflow softmax function.
    """
    x = np.asmatrix(x)
    x = x - np.max(x, axis=1)
    return np.exp(x) / np.sum(np.exp(x), axis=1)


def main():
    pass


if __name__ == '__main__':
    main()
