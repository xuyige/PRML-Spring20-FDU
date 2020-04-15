import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import multivariate_normal
def genDis(mean, cov, pi, size : int):
    """\
    generate data points and point labels.

    Parameters:
    mean -- 2-D array, of shape(C, N)
        Means of the N-dimensional distributions.
    cov -- 3-D array, of shape(C, N, N)
        Covariance matrix of the distributions. All must be symmetric and positive-semidefinite for proper sampling.
    pi -- 1-D array, of length C
        fraction of the result dataset points of each specific class.
    size -- int
        sum of points count to be generated in each distribution.
    """
    mean = np.asarray(mean)
    assert len(mean.shape) == 2
    C, N = mean.shape
    cov = np.asarray(cov)
    assert len(cov.shape) == 3
    assert cov.shape == (C, N, N)
    pi = np.asarray(pi)
    assert pi.size == C
    cnt = (pi / sum(pi) * size).astype(int)
    np.random.seed(123)
    points = np.concatenate([np.random.multivariate_normal(mean[i], cov[i], cnt[i]) for i in range(C)])
    labels = np.concatenate([np.full(cnt[i], i) for i in range(C)])
    GetState = np.random.get_state()
    np.random.shuffle(points)
    np.random.set_state(GetState)
    np.random.shuffle(labels)
    return points, labels

def loadDis(filename = None):
    """\
    returns a disctionary containing the points and the labels
    """
    if filename is None:
        filename = 'dataset.npz'
    return np.load(filename)

class GenerativeModel:
    def __init__(self, C : int, N : int):
        self.C = C
        self.N = N

    def fit(self, points, labels):
        cnt = np.zeros(self.C)
        xsum = np.zeros([self.C, self.N])
        for i in range(labels.size):
            cnt[labels[i]] += 1
            xsum[labels[i]] += points[i]
        pi = cnt / labels.size
        miu = xsum.T / cnt
        miu = miu.T
        # print(miu)
        sigsum = np.zeros([self.C, self.N, self.N])
        for i in range(labels.size):
            sub = points[i] - miu[labels[i]]
            sub = sub[:, np.newaxis]
            sigsum[labels[i]] += sub @ sub.T
        for i in range(self.C):
            sigsum[i] /= cnt[i]
        # print(sigsum)
        return pi, miu, sigsum
        
    def pred(self, p_te: np.ndarray, pi, miu, sigma):
        vars = [multivariate_normal(miu[i], sigma[i]) for i in range(self.C)]
        [multivariate_normal(miu[i], sigma[i]) for i in range(self.C)]
        l_ans = np.empty(p_te.shape[0], int)
        for i in range(l_ans.size):
            l_ans[i] = np.argmax([multivariate_normal.pdf(p_te[i], miu[j], sigma[j]) for j in range(self.C)])
        return l_ans

    @staticmethod
    def test(C, p_tr : np.ndarray, l_tr : np.ndarray, p_te : np.ndarray, l_te : np.ndarray):
        md = GenerativeModel(C, p_tr.shape[1])
        pi, miu, sigma = md.fit(p_tr, l_tr)
        l_ans = md.pred(p_te, pi, miu, sigma)
        p_cr, l_cr = p_te[l_te == l_ans], l_te[l_te == l_ans]
        plt.scatter(p_cr.T[0], p_cr.T[1], c= l_cr, alpha= 0.5) #  cmap= matplotlib.colors.ListedColormap(['r', 'g', 'b']),
        p_er, l_er = p_te[l_te != l_ans], l_te[l_te != l_ans]
        plt.scatter(p_er.T[0], p_er.T[1], c= l_er, marker= 'x', alpha= 0.5)
        print('accuracy = ', l_cr.size / l_te.size)
        plt.show()

class DiscriminativeModel:
    def __init__(self, C : int, N : int):
        self.C = C
        self.N = N

    def fit(self, points : np.ndarray, labels : np.ndarray, alpha = 0.05, maxepoch = 1000, batch = 40):
        W = np.zeros((self.N + 1, self.C))
        I = np.identity(self.C)
        batch = min(batch, labels.size)
        points_ = np.c_[points, np.ones(points.shape[0])]
        epos = 0
        for bps, bls in DiscriminativeModel.minibatch(points_, labels, batch):
            epos += 1
            if epos > maxepoch:
                break
            dW = np.zeros_like(W)
            for i in range(bls.size):
                y_ = DiscriminativeModel.softmax(W.T @ bps[i])
                dy = I[bls[i]] - y_
                dW += bps[i][:, np.newaxis] @ dy[np.newaxis, :]
            W += alpha * dW / bls.size
        # W.T * np.
            if True:
                print(f'Epoch {epos}/{maxepoch}: ', end= '')
                err_p = 0
                for i in range(labels.size):
                    y_ = DiscriminativeModel.softmax(W.T @ points_[i])
                    if np.argmax(y_) != labels[i]:
                        err_p += 1
                print('accuracy:', 1 - err_p / labels.size)
        return W
    
    @staticmethod
    def minibatch(points : np.ndarray, labels : np.ndarray, batch):
        n = labels.size
        i = n
        while True:
            if i == n:
                GetState = np.random.get_state()
                np.random.shuffle(points)
                np.random.set_state(GetState)
                np.random.shuffle(labels)
                i = 0
            j = min(i + batch, n)
            yield points[i:j], labels[i:j]
            i = j

    @staticmethod
    def softmax(x : np.ndarray):
        return np.exp(x) / sum(np.exp(x))

    def pred(self, p_te: np.ndarray, W : np.ndarray):
        p_te_ = np.c_[p_te, np.ones(p_te.shape[0])]
        return [np.argmax(W.T @ p) for p in p_te_]

    @staticmethod
    def test(C, p_tr : np.ndarray, l_tr : np.ndarray, p_te : np.ndarray, l_te : np.ndarray):
        md = DiscriminativeModel(C, p_tr.shape[1])
        W = md.fit(p_tr, l_tr)
        l_ans = md.pred(p_te, W)
        p_te_ = np.c_[p_te, np.ones(p_te.shape[0])]
        # for i in range(l_te.size):
        #     if l_ans[i] != l_te[i]:
        #         print(l_te[i], ':', DiscriminativeModel.softmax(W.T @ p_te_[i]))
        p_cr, l_cr = p_te[l_te == l_ans], l_te[l_te == l_ans]
        plt.scatter(p_cr.T[0], p_cr.T[1], c= l_cr, alpha= 0.5) #  cmap= matplotlib.colors.ListedColormap(['r', 'g', 'b']),
        p_er, l_er = p_te[l_te != l_ans], l_te[l_te != l_ans]
        plt.scatter(p_er.T[0], p_er.T[1], c= l_er, marker= 'x', alpha= 0.5)
        print('accuracy = ', l_cr.size / l_te.size)
        plt.show()

meanA, covA = [2, 2], [[1, 0], [0, 1]]
meanB, covB = [7, 2], [[4, 0], [0, 1]]
meanC, covC = [5, 5], [[1, 0.5], [0.5, 1]]

# meanA, covA = [0, 0], [[0.25, 0], [0, 4]]
# meanB, covB = [0, 0], [[4, 0], [0, 0.25]]
# meanC, covC = [0, 0], [[4, 3], [3, 4]]

means = [meanA, meanB, meanC]
covs = [covA, covB, covC]
trainratio = 0.8
if __name__ == '__main__':
    points, labels = genDis(means, covs, [1/3, 1/3, 1/3], 5000)
    plt.scatter(points.T[0], points.T[1], c= labels, alpha= 0.5)
    plt.show()
    np.savez('dataset', points, labels, points=points, labels=labels)
    # loadDis()
    sp = int(labels.size * trainratio)
    points_train, points_test = points[:sp], points[sp:]
    labels_train, labels_test = labels[:sp], labels[sp:]
    GenerativeModel.test(len(means), points_train, labels_train, points_test, labels_test)
    DiscriminativeModel.test(len(means), points_train, labels_train, points_test, labels_test)