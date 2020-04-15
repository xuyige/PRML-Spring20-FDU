#!/usr/bin/env python3

#
# See:
#     ./source.py --help
#     ./source.py generate --help
#     ./source.py run --help
#


import sys
import argparse

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt


# Pyplot parameters

T = 512
X_VIEW = (-1, 1)
Y_VIEW = (-1, 1)


# Utilities

def generate_data(n, prior, center, scale):
    """
    @parameter
    n : int
        size of dataset.
    prior : list[float]
        priors for each class. This would be normalized before applying.
    center : list[vec]
        centers for the normal distribution of each class.
    scale : list[float]
        variance for the normal distribution of each class.

    @return
    (x, t, v)
    x : list[vec]
        data points.
    t : list[int]
        type of each point.
    y : list[vec]
        one-hot vectors.

    @example
    ```
    >>> generate_data(100, [1, 1, 1],
            [[0.1, 0.1], [0.4, 0.9], [0.9, 0.4]],
            [0.15, 0.09, 0.12]
        )
    ```
    """

    C, K = len(prior), len(center[0])
    prior = np.array(prior) / sum(prior)  # normalize
    t = np.random.choice(np.r_[:C], n, p=prior)
    t.sort()

    ti, cnt = np.unique(t, return_counts=True)
    x = np.concatenate([
        np.random.normal(center[i], scale[i], (c, K))
        for i, c in zip(ti, cnt)
    ])
    y = np.identity(C)[t]

    return t, x, y

def generate_grid(x_view, y_view, n):
    """
    @parameter
    x_view : (x_min: float, x_max: float)
        view of x coordinate.
    y_view : (y_min: float, y_max: float)
        view of y coordinate.
    n : int
        number of samples in both x and y direction.

    @return
    (X, Y, P)
    X, Y
        returned by `np.meshgrid`.
    P : list[vec]
        list of all vectors.
    """

    X = np.linspace(*X_VIEW, T)
    Y = np.linspace(*Y_VIEW, T)
    X, Y = np.meshgrid(X, Y)
    P = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

    return X, Y, P

def evaluate_grid(f, x_view, y_view, n):
    """
    Evaluate function on a grid.

    See `generate_grid`.
    """

    X, Y, P = generate_grid(X_VIEW, Y_VIEW, T)
    return X, Y, f(P)

def scatter(t, x):
    """
    Scatter dataset.

    @parameter
    t : list[int]
        labels.
    x : list[vec]
        data points.
    """

    L = 'ABC'
    for i in range(3):
        plt.scatter(
            x[t == i][:, 0], x[t == i][:, 1],
            alpha=0.4, label=L[i]
        )
    plt.legend()

def test(f, t, x):
    """
    Test model

    @parameter
    f : function
        `model.predicts`.
    t : list[int]
        labels.
    x : list[vec]
        data points.

    @return : float
        accuracy (value in [0, 1])
    """

    Z = f(x)
    cnt = np.count_nonzero(Z == t)
    return cnt / len(t)

def test_and_print(f, t, x):
    print('Accuracy: %.2f%%' % (100 * test(f, t, x)))


class GenerativeModel:
    def __init__(self, C, K):
        """
        @parameter
        C : int
            number of classes.
        K : int
            number of dimensions.
        """

        self.C = C
        self.K = K
        self.reset()

    def reset(self):
        self.center = np.zeros((self.C, self.K))
        self.snorm = np.zeros(self.C)
        self.variance = np.zeros(self.C)
        self.count = np.zeros(self.C, dtype=int)

    def fit(self, t, x):
        """
        @parameter
        t : list[int]
            type array.
        x : list[vec]
            data points.

        @remark
            this is an online algorithm. Applying multiple `fit`s is feasible.
        """

        def norm2(u):
            return (u**2).sum()

        ti, cnt = np.unique(t, return_counts=True)
        for i, n in zip(ti, cnt):
            xs = x[t == i]
            self.center[i] = self.center[i] * self.count[i] + xs.sum(axis=0)
            self.count[i] += n
            self.center[i] /= self.count[i]

            self.snorm[i] += norm2(xs)
            self.variance[i] = 0.5 * (
                self.snorm[i] / self.count[i] - norm2(self.center[i]))

    def predicts(self, xs):
        """
        Give predictions based on fitted model.

        @parameter
        xs : list[vec]
            positions to be predicted.

        @return
        list[int]
            predicted classes.
        """

        prior = self.count / self.count.sum()
        offset = ((xs[:, np.newaxis, :] - self.center)**2).sum(axis=2)
        likelihood = np.exp(-offset / 2 / self.variance) / self.variance
        posterior = likelihood * prior
        return np.argmax(posterior, axis=1)

    def predict(self, x):
        """
        Shortcut for `predicts`.
        """

        return self.predicts([x])[0]


class DiscriminativeModel:
    def __init__(self, C, K):
        """
        @parameter
        C : int
            number of classes.
        K : int
            number of dimensions.
        """

        self.C = C
        self.K = K
        self.reset()

    def reset(self):
        self.weight = np.zeros((self.K + 1, self.K + 1))

    def _fit(self, y, x, k=1., eps=1e-6, max_iter=200, show_steps=False):
        """
        [deprecated]

        @parameter
        y : list[vec]
            labels in one-hot encoding.
        x : list[vec]
            data points.
        k = 1 : float
            constant learning rate.
        eps = 1e-6 : float
            destined precision.
        max_iter = 200 : int
            maximum number of iterations.
        show_steps = False : bool
            whether to show debug information at the end of each iteration.

        @remark
            This is stochastic gradient descent algorithm.
        """

        def _descent(i):
            ex = np.concatenate((x[i], [1])).reshape(-1, 1)
            v = sp.softmax(self.weight.T @ ex)
            self.weight += k * (ex @ (y[i].reshape(-1, 1) - v).T)

        descent = np.vectorize(_descent)

        W0 = self.weight.copy()
        idx = np.r_[:len(x)]
        for j in range(max_iter):
            np.random.shuffle(idx)
            descent(idx)

            delta = np.sqrt(((W0 - self.weight)**2).sum())
            if show_steps:
                print(f'[{j + 1}] {delta}')
            if delta < eps:
                if show_steps:
                    print(f'[{j + 1}] converaged.')
                break
            else:
                W0 = self.weight.copy()

    def _batch_descent(self, y, x, k):
        dy = y[:, np.newaxis, :] - sp.softmax(x @ self.weight, axis=2)
        self.weight += k * (x.reshape(-1, 3, 1) @ dy).sum(axis=0)

    def fit(
        self, y, x,
        test_data=None, batch=64,
        k0=1., d=0,
        t0=1., t_rate=0.85,
        max_iter=1000, show_steps=False
    ):
        """
        @parameter
        y : list[vec]
            labels in one-hot encoding.
        x : list[vec]
            data points
        test_data = None : (t: list[int], x: list[vec])
            test data for early stopping. `None` to disable early stopping.
        batch = 64 : int
            number of data points to be processed per batch.
        k0 = 1 : float
            initial learning rate.
        d = 0 : float
            decay parameter for learning rate.
            d = 0 disable learning rate decaying.
        t0 = 1 : float
            initial tolerance parameter.
        t_rate = 0.85 : float
            tolerance parameter decay parameter.
        max_iter = 1000 : int
            maximum number of iterations.
        show_steps = False : bool
            whether to show debug outputs.
        """

        n = len(x)
        x = np.hstack((x, np.ones((n, 1))))[:, np.newaxis, :]
        idx = np.r_[:n]

        p0, W0 = 0, None
        i = n
        for j in range(max_iter):
            if i >= n:
                np.random.shuffle(idx)
                i = 0

            r = min(i + batch, n)
            self._batch_descent(y[idx[i:r]], x[idx[i:r]], k0)

            if show_steps:
                print(f'[{j + 1}] interval [{i}:{r})')
            i = r
            k0 /= 1 + d * (j + 1)

            if test_data:
                p = test(self.predicts, *test_data)

                if show_steps:
                    print(f'[{j + 1}] accuracy = {p}, t0 = {t0}')

                if p > p0:
                    p0 = p
                    W0 = self.weight.copy()
                if p0 - p > t0:
                    if show_steps:
                        print('stopped')
                    break

                t0 *= t_rate

        self.weight = W0

    def predicts(self, xs):
        """
        Give predictions based on fitted model.

        @parameter
        xs : list[vec]
            positions to be predicted.

        @return
        list[int]
            predicted classes.
        """

        xs = np.hstack((xs, np.ones((len(xs), 1))))
        result = xs[:, np.newaxis, :] @ self.weight
        return np.argmax(result, axis=2).ravel()

    def predict(self, x):
        """
        Shortcut for `predicts`.
        """

        return self.predicts([x])[0]


# Main

class LabelParams:
    def __init__(self, params: str):
        L = params.split(',')
        if len(L) != 4:
            raise ValueError(f'invalid format: "{params}". Should be "x,y,scale,weight"')

        self.x, self.y, self.scale, self.weight = map(float, L)

    def __repr__(self):
        return f'({self.x}, {self.y}) {self.scale} [{self.weight}]'

def generate(args):
    a, b, c = args.a, args.b, args.c
    params =[
        [a.weight, b.weight, c.weight],
        [[a.x, a.y], [b.x, b.y], [c.x, c.y]],
        [a.scale, b.scale, c.scale]
    ]

    t, x, _ = generate_data(args.n, *params)
    z = np.hstack((x, t[:, np.newaxis]))
    np.savetxt(args.output, z)

    if args.show:
        plt.xlim(X_VIEW)
        plt.ylim(Y_VIEW)
        scatter(t, x)
        plt.show()

def run(args):
    def split(z):
        t, x = z[:, 2].reshape(-1).astype(np.int), z[:, :2]
        y = np.identity(args.c)[t]
        return t, x, y

    z = np.loadtxt(args.dataset)
    n = len(z)

    np.random.shuffle(z)
    n1 = int(n * args.percent)
    n0 = n - n1
    n3 = int(n0 * args.percent)
    n2 = n0 - n3
    t0, x0, y0 = split(z[:n0])  # dataset for g-model
    t1, x1, y1 = split(z[n0:])  # testset
    t2, x2, y2 = split(z[:n2])  # dataset for d-model
    t3, x3, y3 = split(z[n2:n0])  # testset for d-model

    plt.gcf().set_size_inches((11, 5))

    model1 = GenerativeModel(args.c, args.k)
    model1.fit(t0, x0)
    plt.subplot(121)

    print('shared testset size:', len(t1), '\n')

    print('# Generative Model:')
    print('\tdataset size:', len(t0))
    print('\tcenter:', model1.center)
    print('\tvariance:', model1.variance)
    print('\tcount:', model1.count)
    test_and_print(model1.predicts, t1, x1)

    plt.xlim(X_VIEW)
    plt.ylim(Y_VIEW)
    plt.title('Generative Model')
    X, Y, Z = evaluate_grid(model1.predicts, X_VIEW, Y_VIEW, T)
    plt.contourf(X, Y, Z.reshape(T, T), alpha=0.1)
    scatter(t1, x1)

    print('')

    model2 = DiscriminativeModel(args.c, args.k)
    model2.fit(
        y2, x2, test_data=(t3, x3),
        t_rate=args.tr, d=args.d, batch=args.batch,
        show_steps=args.verbose
    )
    plt.subplot(122)

    print('# Discriminative Model:')
    print('\tdataset size:', len(t2))
    print('\ttestset size:', len(t3))
    print('\tweight:', model2.weight)
    test_and_print(model2.predicts, t1, x1)

    plt.xlim(X_VIEW)
    plt.ylim(Y_VIEW)
    plt.title('Discriminative Model')
    X, Y, Z = evaluate_grid(model2.predicts, X_VIEW, Y_VIEW, T)
    plt.contourf(X, Y, Z.reshape(T, T), alpha=0.1)
    scatter(t1, x1)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument('-c', default=3, type=int, help='number of classes. Default: 3')
    parser.add_argument('-k', default=2, type=int, help='number of dimensions. Default: 2')

    subparsers = parser.add_subparsers(required=True, title='subcommands', help='all available subcommands', dest='subcommand')
    gp = subparsers.add_parser('generate', help='generate data')
    gp.add_argument('-s', '--show', action='store_true', help='show the generated dataset with pyplot')
    gp.add_argument('n', metavar='N', type=int, help='dataset size')
    gp.add_argument('-o', '--output', default='a.data', type=str, help='path to save data file. Default: "a.data"')

    labels = 'ABC'
    for X in labels:
        gp.add_argument(f'-{X.lower()}',
            default=LabelParams('0,0,0.2,1'), type=LabelParams,
            help=f'parameters "x,y,scale,weight" for label {X}'
        )

    rp = subparsers.add_parser('run', help='run model fitting and testing')
    rp.add_argument('dataset', metavar='DATA', help='data file for training & testing')
    rp.add_argument('-p', '--percent', default=0.2, type=float, help='percent of dataset will be used as test data. Default: 20%%')
    rp.add_argument('-tr', default=0.85, type=float, help='SGD parameter `t_rate`. Default: 0.85')
    rp.add_argument('-d', default=0.0, type=float, help='SGD parameter `d`. Default: 0')
    rp.add_argument('-b', '--batch', default=64, type=int, help='SGD parameter `batch_size`. Default: 64')
    rp.add_argument('-v', '--verbose', action='store_true', help='show SGD traces')

    args = parser.parse_args(sys.argv[1:])
    locals()[args.subcommand](args)
