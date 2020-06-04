import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, multivariate_normal

from sklearn import metrics
from sklearn.metrics import davies_bouldin_score


#################### Generate Data ####################
def generate_data(n_data, means, covariances, weights):

    n_clusters, n_features = means.shape
    labels=[]

    data = np.zeros((n_data, n_features))
    for i in range(n_data):
        # pick a cluster id and create data from this cluster
        k = np.random.choice(n_clusters, size=1, p=weights)[0]
        x = np.random.multivariate_normal(means[k], covariances[k])
        data[i] = x
        labels.append(k)

    return data,labels


#################### GMM Model ####################
class GMM:
    def __init__(self, n_components: int, n_iters: int, tol: float, seed: int):
        self.n_components = n_components
        self.n_iters = n_iters
        self.tol = tol
        self.seed = seed

    def fit(self, X):
        # data's dimensionality and responsibility vector
        n_row, n_col = X.shape
        self.resp = np.zeros((n_row, self.n_components))

        # initialize parameters
        np.random.seed(self.seed)
        chosen = np.random.choice(n_row, self.n_components, replace=False)
        self.means = X[chosen]
        self.weights = np.full(self.n_components, 1 / self.n_components)

        # for np.cov, rowvar = False,indicates that the rows represents obervation
        shape = self.n_components, n_col, n_col
        self.covs = np.full(shape, np.cov(X, rowvar=False))

        log_likelihood = 0
        self.converged = False
        self.log_likelihood_trace = []
        self.labels=[]

        for i in range(self.n_iters):
            log_likelihood_new = self._do_estep(X)
            self._do_mstep(X)
            self.get_labels()

            if abs(log_likelihood_new - log_likelihood) <= self.tol:
                self.converged = True
                break

            log_likelihood = log_likelihood_new
            self.log_likelihood_trace.append(log_likelihood)

        return self

    def _do_estep(self, X):

        self._compute_log_likelihood(X)
        log_likelihood = np.sum(np.log(np.sum(self.resp, axis=1)))

        # normalize over all possible cluster assignments
        self.resp = self.resp / self.resp.sum(axis=1, keepdims=1)
        return log_likelihood

    def _compute_log_likelihood(self, X):
        for k in range(self.n_components):
            prior = self.weights[k]
            likelihood = multivariate_normal(self.means[k], self.covs[k]).pdf(X)
            self.resp[:, k] = prior * likelihood

        return self

    def _do_mstep(self, X):

        # total responsibility assigned to each cluster, N^{soft}
        resp_weights = self.resp.sum(axis=0)

        # weights
        self.weights = resp_weights / X.shape[0]

        # means
        weighted_sum = np.dot(self.resp.T, X)
        self.means = weighted_sum / resp_weights.reshape(-1, 1)

        # covariance
        for k in range(self.n_components):
            diff = (X - self.means[k]).T
            weighted_sum = np.dot(self.resp[:, k] * diff, diff.T)
            self.covs[k] = weighted_sum / resp_weights[k]

        return self

    def get_labels(self):

        self.labels.clear()
        for row in self.resp:
            max_index=0
            for j in range(0,self.n_components):
                if(row[j]>row[max_index]):
                    max_index=j
            self.labels.append(max_index)


#################### Evaluation ####################
def evaluation(X,labels):
    SC=metrics.silhouette_score(X, labels, metric='euclidean')
    CHI=metrics.calinski_harabasz_score(X, labels)
    DBI=davies_bouldin_score(X, labels)

    return round(SC,4),round(CHI,4),round(DBI,4)

def AIC_BIC(X):
    n_components = np.arange(1, 10)
    clfs = [GaussianMixture(n, max_iter=1000).fit(X) for n in n_components]
    bics = [clf.bic(X) for clf in clfs]
    aics = [clf.aic(X) for clf in clfs]

    plt.plot(n_components, bics, label='BIC')
    plt.plot(n_components, aics, label='AIC')
    plt.xlabel('n_components')
    plt.legend()
    plt.show()


#################### Plot ####################
def plot_gaussian_distribution(means,covariances):

    x, y = np.mgrid[-4:4:.01, -4:4:.01]
    position = np.empty(x.shape + (2,))
    position[:, :, 0] = x
    position[:, :, 1] = y

    # different values for the covariance matrix
    titles = ['spherical', 'diag', 'full']

    plt.figure(figsize=(15, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        z = multivariate_normal(means[i], covariances[i]).pdf(position)
        plt.contour(x, y, z)
        plt.title('{},  {}'.format(titles[i], covariances[i]))
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.xticks([])
        plt.yticks([])

    plt.show()

def plot_contours(X, Y, means, covs, title):
    """visualize the gaussian components over the data"""

    col = ['green', 'red', 'indigo', 'yellow']
    clusters=[]
    plt.figure()

    for i in range(0,len(set(Y))):
        clusters.append([X[j] for j in range(0,len(X)) if Y[j]==i])
        clusters[i]=np.array(clusters[i])

        plt.plot(clusters[i][:,0],clusters[i][:,1],'ko',markersize=2,color=col[i])

    delta = 0.025
    k = means.shape[0]

    x = np.arange(-4.0, 8.0, delta)
    y = np.arange(-5.0, 12.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)

        C=plt.contour(x_grid, y_grid, z_grid,colors=col[i])
        # plt.clabel(C, inline=True, fontsize=8)

    # plt.xticks(())
    # plt.yticks(())
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    # generate data
    init_means = np.array([
        [5, 0],
        [1, 1],
        [0, 5]
    ])
    init_covariances = np.array([
        [[1,0],[0,1]],
        [[1,0],[0,3]],
        [[1,-1],[-1,3]]
    ])
    init_weights = [1 / 4, 1 / 2, 1 / 4]

    np.random.seed(4)
    X, Y = generate_data(300, init_means, init_covariances, init_weights)

    # model
    seed = 4
    gmm = GMM(n_components=3, n_iters=1, tol=1e-4, seed=seed)
    gmm.fit(X)
    Y = gmm.labels
    plot_contours(X,Y, gmm.means,  gmm.covs, 'Initial clusters')

    gmm = GMM(n_components=3, n_iters=500, tol=1e-4, seed=seed)
    gmm.fit(X)
    Y = gmm.labels
    plot_contours(X, Y, gmm.means, gmm.covs, 'Final clusters')

    # evaluate
    print('converged iteration:', len(gmm.log_likelihood_trace))
    SC, CHI, DBI = evaluation(X, Y)
    # AIC_BIC(X)
    print("Silhouette Coefficient:",SC)
    print("Calinski-Harabasz Index:",CHI)
    print("Davies-Bouldin Index:",DBI)


if __name__=='__main__':
    main()




