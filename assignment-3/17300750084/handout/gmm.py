import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def plot_elipse(mu, cov, lc, ax, ls=':'):
    vals, vecs = np.linalg.eigh(cov)
    dx, dy = vecs[:, 1]
    theta = np.degrees(np.arctan2(dy, dx))
    h, w = np.sqrt(5.991 * vals)
    ax[0].add_patch(Ellipse(mu, width=w, height=h, angle=theta,
                         edgecolor=lc, alpha=1.0, fc=None, lw=2, fill=None, ls=ls))


def multivariate_normal(data, mean, var, d):
    cov = np.asmatrix(var)
    inverse_cov = cov.I
    p1 = 1 / (np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(cov)))
    p = []
    mean = np.asarray(mean)
    for x in data:
        x = np.asarray(x)
        pi = p1 * np.exp(-1 / 2 * np.dot(np.transpose(x - mean), inverse_cov).dot(x - mean))
        pi = pi.reshape(1,)
        if p == []:
            p = pi
        else:
            p = np.concatenate((p, pi))
    return p.reshape(-1)

def plot_raw_date2d(data, label, n_clusters, mu, var):
    color_list = ['#8ae9bd', '#fba383', '#9cdadf','#ee92d9']
    ax = plt.gca()
    for j in range(n_clusters):
        labeli = np.where(label == j)
        xi = data[labeli]
        plt.scatter(x= xi[:, 0], y=xi[:, 1], c=color_list[j], s= 3)
        plot_elipse(mu[j], var[j], color_list[j], [ax], '--')
    plt.show()


class gmm():
    def __init__(self, n_clusters, n_samples, n_dim, rand_mu):
        self.k = n_clusters
        self.n = n_samples
        self.d = n_dim
        self.mu = rand_mu
        self.var = np.asarray([np.diag(np.ones(self.d)) for i in range(self.k)])
        self.px = np.ones((self.n, self.k), dtype=int) / self.k
        self.pz = np.sum(self.px, axis=0) / np.sum(self.px)

    def init_with_kmeans(self, n_iter, x):
        self.mu, self.var = self.run_Kmeans(n_iter, x)



    def EM(self, x):
        pi = np.zeros((self.n, self.k))
        for i in range(self.k):
            pi[:, i] = self.pz[i] * multivariate_normal(x, self.mu[i], self.var[i], self.k)
        self.px = pi / np.sum(pi, axis=1).reshape(-1, 1)
        self.pz = np.sum(self.px, axis=0) / np.sum(self.px)

        # M
        for i in range(self.k):
            self.mu[i] = np.average(x, weights=self.px[:,i], axis = 0)
            self.var[i] = np.average(np.asarray([np.dot(np.asmatrix(xx - self.mu[i]).T,np.asmatrix(xx - self.mu[i])) for xx in x]), weights=self.px[:,i], axis=0)
        logp = np.log(np.sum(pi, axis=1))
        print(logp)

    def run_Kmeans(self, n_iter, x):
        center_list = x[np.random.choice(range(self.n), self.k)]
        var_list = np.ones((self.k, self.d, self.d))
        res = np.ones(self.n)
        for i in range(n_iter):
            for j in range(self.n):
                min_dist = np.infty
                min_id = -1
                for id, c in enumerate(center_list):
                    if np.sqrt(np.sum((x[j] - c)**2)) < min_dist:
                        min_id = id
                        min_dist = np.sqrt(np.sum((x[j] - c)**2))
                res[j] = min_id
            for id, c in enumerate(center_list):
                cat_list = x[np.where(res==id)]
                center_list[id] = np.average(cat_list, axis=0)
                var_list[id] = np.cov(cat_list.transpose())

        ax = plt.gca()
        color_list = ['#8ae9bd', '#fba383', '#9cdadf', '#ee92d9']
        for j in range(self.k):
            labeli = np.where(res == j)
            xi = x[labeli]
            plot_elipse(center_list[j], var_list[j],'#808080',[ax])
            plt.scatter(x=xi[:, 0], y=xi[:, 1], c=color_list[j], s=3)
        plt.show()

        return center_list, var_list


    def run_EM(self, n_iter, x,  real_mean, real_var):
        k_center, k_var= self.run_Kmeans(n_iter, x)

        for i in range(n_iter):
            self.EM(x)

            if i % 10 == 0:
                color_list = ['#8ae9bd', '#fba383', '#9cdadf','#ee92d9']
                p_label = np.argmax(self.px, axis=1)
                ax = plt.gca()
                for j in range(self.k):
                    labeli = np.where(p_label == j)
                    xi = x[labeli]
                    plot_elipse(real_mean[j], real_var[j], '#808080', [ax])
                    plot_elipse(self.mu[j], self.var[j],'#C71585', [ax])
                    plt.scatter(x=xi[:, 0], y=xi[:, 1], c=color_list[j], s=3)
                plt.show()
            if i  == n_iter - 1:
                color_list = ['#8ae9bd', '#fba383', '#9cdadf','#ee92d9']
                p_label = np.argmax(self.px, axis=1)
                ax = plt.gca()
                for j in range(self.k):
                    labeli = np.where(p_label == j)
                    xi = x[labeli]
                    plot_elipse(real_mean[j], real_var[j], '#808080', [ax])
                    plot_elipse(self.mu[j], self.var[j], '#C71585', [ax])
                    plot_elipse(k_center[j], k_var[j],'#4D80E6', [ax] )
                    plt.scatter(x=xi[:, 0], y=xi[:, 1], c=color_list[j], s=3)
                plt.show()





