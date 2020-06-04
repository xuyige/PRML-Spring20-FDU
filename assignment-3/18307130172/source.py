from typing import List, Tuple

import argparse
import itertools

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

Vector = Tuple[float, float]
Covariance = List[List[float]]

PRIORS = [0.3, 0.2, 0.4, 0.1]
MEANS = [
    (-0.5, -0.6),
    (0.6, -0.4),
    (0.1, 0.3),
    (0.0, -0.3)
]
COVS = [[  # must be symmetric & positive-semidefinite
    [0.03, 0],
    [0, 0.03]
], [
    [0.04, -0.02],
    [-0.02, 0.05]
], [
    [0.07, 0.04],
    [0.04, 0.06]
], [
    [0.005, 0],
    [0, 0.005]
]]

def plot_data(data: np.ndarray, **kwargs):
    plt.scatter(data[:, 0], data[:, 1], **kwargs)

def generate_data(
    n: int, prior: List[float],
    mean: List[Vector],
    cov: List[Covariance]
) -> np.ndarray:
    cnt = np.random.multinomial(n, prior)
    data = np.vstack([
        np.hstack((
            np.random.multivariate_normal(
                mean[i], cov[i], cnt[i]
            ),
            i * np.ones(cnt[i]).reshape(-1, 1)
        ))
        for i in range(len(prior))
    ])
    np.random.shuffle(data)
    return data

def get_label_by_distance(
    data: np.ndarray,
    p: np.ndarray,
    k: int
) -> np.ndarray:
    dist = np.hstack([
        ((data - p[i])**2).sum(axis=1).reshape(-1,1)
        for i in range(k)
    ])
    label = dist.argmin(axis=1)
    return label

def k_means(
    data: np.ndarray,
    k: int, steps: int,
    debug=False
) -> np.ndarray:
    # p = np.random.normal(scale=0.5, size=(k, 2))
    # choose data points to avoid NaN.
    p = data[np.random.choice(len(data), k)]

    for _ in range(steps):
        label = get_label_by_distance(data, p, k)

        if debug:
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)

            for i in range(k):
                plot_data(data[label == i], alpha=0.25)
            plot_data(p, marker='x')
            plt.show()

        for i in range(k):
            p[i] = np.average(data[label == i], axis=0)

    return p

def get_label_by_guassian(
    data: np.ndarray,
    pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    ds = [
        sp.stats.multivariate_normal(mu[i], sigma[i])
        for i in range(k)
    ]  # distributions
    prod = np.hstack([
        (pi[i] * ds[i].pdf(data)).reshape(-1, 1)
        for i in range(k)
    ])
    label = prod.argmax(axis=1)
    return prod, label

def expectation_maximize(
    data: np.ndarray,
    k: int, k_steps: int, e_steps: int,
    debug=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # use k-means algorithm to initialize
    mu = k_means(data, k, k_steps, debug=debug)
    label = get_label_by_distance(data, mu, k)
    _, cnt = np.unique(label, return_counts=True)
    pi = cnt / cnt.sum()
    sigma = np.array([
        np.diag(np.var(data[label == i], axis=0))
        for i in range(k)
    ])

    for _ in range(e_steps):
        # evaluate posterior probabilities
        prod, label = get_label_by_guassian(data, pi, mu, sigma, k)
        gamma = prod / prod.sum(axis=1).reshape(-1, 1)

        if debug:
            elbo = (gamma * np.log(prod / gamma)).sum()
            print(f'ELBO = {elbo}')

            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            for i in range(k):
                plot_data(data[label == i], alpha=0.25)
            plot_data(mu, marker='x')
            plt.show()

        # update parameters
        cnt = gamma.sum(axis=0)
        pi = cnt / cnt.sum()
        mu = (gamma.transpose() @ data) / cnt.reshape(-1, 1)

        # broadcasting is evil...
        diff = data[np.newaxis, ...] - mu[:, np.newaxis, :]
        diff = diff[..., np.newaxis]
        result = (diff @ diff.transpose(0, 1, 3, 2)).transpose(1, 0, 2, 3)
        sigma = result * gamma[..., np.newaxis, np.newaxis]
        sigma = sigma.sum(axis=0) / cnt[:, np.newaxis, np.newaxis]

    return pi, mu, sigma

def cli_generate(args):
    data = generate_data(args.number, PRIORS, MEANS, COVS)
    with open(args.output, 'w') as fp:
        np.savetxt(fp, data[:, 0:2])
    with open(f'{args.output}.label', 'w') as fp:
        np.savetxt(fp, data[:, 2], fmt='%.0f')

def cli_run(args):
    with open(args.data, 'r') as fp:
        data = np.loadtxt(fp)

    pi, mu, sigma = expectation_maximize(
        data, args.K, args.kstep, args.estep, args.verbose
    )
    _, label = get_label_by_guassian(data, pi, mu, sigma, args.K)

    print(f'μ = {mu}')
    print(f'π = {pi}')

    if args.accuracy:
        with open(f'{args.data}.label', 'r') as fp:
            std = np.loadtxt(fp, dtype=np.int)

        # since label name may be different, enumerate all
        # possible remappings to get the right accuracy.
        accuracy = 0
        for perm in itertools.permutations(range(args.K)):
            remap = label.copy()
            for i in range(args.K):
                remap[label == i] = perm[i]

            result = std == remap
            val = np.count_nonzero(result) / len(result)
            accuracy = max(accuracy, val)

        print(f'Accuracy: {accuracy * 100}%')

    if args.plot:
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        for i in range(args.K):
            plot_data(data[label == i], alpha=0.25)
        plot_data(mu, marker='x', color='red')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    generate_parser = subparser.add_parser('generate', help='Generate data.')
    generate_parser.set_defaults(handler=cli_generate)
    generate_parser.add_argument(
        '-n', '--number', type=int, default=100,
        help='Number of data points.'
    )
    generate_parser.add_argument(
        '-o', '--output', type=str, default='fake.data',
        help='File where data will be saved.'
    )

    run_parser = subparser.add_parser('run', help='Run EM algorithm.')
    run_parser.set_defaults(handler=cli_run)
    run_parser.add_argument('data', type=str, help='File path of the data.')
    run_parser.add_argument('-k', '--kstep', type=int, default=10, help='Number of steps for K-means algorithm.')
    run_parser.add_argument('-e', '--estep', type=int, default=30, help='Number of steps for EM algorithm.')
    run_parser.add_argument('-v', '--verbose', action='store_true', help='Show steps.')
    run_parser.add_argument('-p', '--plot', action='store_true', help='Show result wuth pyplot.')
    run_parser.add_argument('-a', '--accuracy', action='store_true', help='Calculate accuracy.')
    run_parser.add_argument('-K', type=int, default=3, help='Number of clusters.')

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
