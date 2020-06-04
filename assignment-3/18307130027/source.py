import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.colors
import itertools
Prior = [0.5, 0.2, 0.3]
Mean = [[-0.4, -0.35], [0, 0.3], [0.5, 0]]
Cov = [[
    [0.01, 0],
    [0, 0.01]
], [
    [0.08, 0],
    [0, 0.08]
], [
    [0.01, 0],
    [0, 0.01]
]
]
def gen_data(N, prior = Prior, mean = Mean, cov = Cov):
    n = len(mean)
    d = len(mean[0])
    labelcnt = np.random.multinomial(N, prior)
    data = np.vstack([
        np.random.multivariate_normal(mean[i], cov[i], labelcnt[i])
        for i in range(n)
    ])
    label = [np.full(labelcnt[i], i) for i in range(n)]
    label = np.concatenate(label)
    idx = np.random.permutation(N)
    return data[idx], label[idx]

def save_data(data, label):
    np.savetxt('data.data', data)
    np.savetxt('label.data', label)

def load_data():
    data = np.loadtxt('data.data')
    label = np.loadtxt('label.data', np.int32)
    return data, label

def plot_(data, label):
    plt.scatter(data[:,0], data[:,1], c= label, cmap=matplotlib.colors.ListedColormap(['r', 'g', 'b']), alpha= 0.5)
    plt.show()

def plot_km(data, label, glabel):
    rights = label == glabel
    wrongs = label != glabel
    plt.scatter(data[rights,0], data[rights,1], marker = '.', c= label[rights], cmap=matplotlib.colors.ListedColormap(['r', 'g', 'b']), alpha=0.5)
    plt.scatter(data[wrongs,0], data[wrongs,1], marker = 'x', c= label[wrongs], cmap=matplotlib.colors.ListedColormap(['r', 'g', 'b']), alpha=0.5)
    plt.show()

def plot_em(data, em_result, glabel):
    label = np.argmax(em_result, axis = 1)
    rights = label == glabel
    wrongs = label != glabel
    plt.scatter(data[rights,0], data[rights,1], marker = '.', c= em_result[rights, :], alpha=0.5)
    plt.scatter(data[wrongs,0], data[wrongs,1], marker = 'x', c= em_result[wrongs, :], alpha=0.5)
    plt.show()
    pass

def kmeans_train(data, k, *, itermax = 100, m = None):
    if m is None:
        m = data[np.random.choice(len(data), k)]
    d = data.shape[1]
    for i in range(itermax):
        if i % 10 == 0:
            yield m
        sump = np.zeros((k, d))
        cnt = np.zeros(k, np.int32)
        for pt in data:
            dis = np.linalg.norm(m - pt, axis = 1)
            lb = np.argmin(dis)
            sump[lb] += pt
            cnt[lb] += 1
        sump /= cnt[:, None]
        # if (np.allclose(m, sump)):
        #     return sump
        m = sump # new mean

    # return sump

def kmeans_infer(data, mean):
    label = np.empty(len(data), np.int32)
    for i in range(len(data)):
        label[i] = np.argmin(np.linalg.norm(mean - data[i], axis = 1))

    return label

def em_train(data, k, labeli, itermax = 100):
    gamma = np.eye(k)[labeli]
    n = len(data)
    for i in range(itermax):
        N = np.sum(gamma, axis = 0) # (k, )
        pi = N / n # (k, )
        miu = gamma[:, :, None] * data[:, None, :]
        miu = np.sum(miu, axis = 0)
        miu /= N[:, None]
        det = data[:, None, :] - miu[None, :, :]
        det = det[..., None] @ det[..., None, :]
        sigma = gamma[:, :, None, None] * det
        sigma = np.sum(sigma, axis = 0)
        sigma /= N[:, None, None]
        if i % 10 == 0:
            yield pi, miu, sigma
        # E
        gamma = em_infer(data, pi, miu, sigma)
    
    # return pi, miu, sigma

def em_infer(data, pi, miu, sigma):
    n, d = data.shape
    k = len(pi)
    res = np.empty((n, k))
    ds = [multivariate_normal(miu[i], sigma[i]) for i in range(k)]
    for i in range(n):
        res[i] = [pi[j] * ds[j].pdf(data[i]) for j in range(k)]
    res /= np.sum(res, axis = 1)[:, None]
    return res

if __name__ == '__main__': 
    # tmp = gen_data(1000)
    # print(tmp)
    # save_data(tmp[0], tmp[1])
    tmp = load_data()
    plot_(tmp[0], tmp[1])
    k = len(Mean)
    data, Glabel = tmp
    n = len(data)
    # kmResult = kmeans_train(data, k)
    # plot_(data, kmeans_label)
    for kmResult in kmeans_train(data, k):
        # plot_(data, kmeans_infer(data, kmResult))
        kmeans_label = kmeans_infer(data, kmResult)
        maxaccuracy = 0
        for f in itertools.permutations(range(k)): # try to sync label
            testf = np.array(f)
            newac = np.count_nonzero(testf[kmeans_label] == Glabel) / n
            if maxaccuracy < newac:
                maxaccuracy = newac
                truef = testf
        print(f'KMaccuracy: {maxaccuracy}')

    kmeans_label = truef[kmeans_label]
    plot_km(data, kmeans_label, Glabel)

    for pi, miu, sigma in em_train(data, k, kmeans_label):
        em_result = em_infer(data, pi, miu, sigma)
        label = np.argmax(em_result, axis = 1)
        accuracy = np.count_nonzero(label == Glabel) / n
        print(f'EMaccuracy: {accuracy}')
        # loss = np.sum(np.log(np.eye(k)[Glabel] / em_result))
        # print(f'EMloss: {loss}')
        
        
    plot_em(data, em_result, Glabel)