import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.stats import multivariate_normal

def generate_data(n=1000, mu=None, sigma=None, prior_p=None):
    '''
    generate_data.

    :param mu:
        $\vec{\mu}$
    :param sigma:
        $\Sigma$
    :param prior_p:
        $prior probability$
    :param n:
        # of samples
    :return:
        sample, label
    '''
    if mu is None:
        mu = [[2,2],[0,-2],[-2,2]]
    if sigma is None:
        sigma = [[1,0],[0,1]]
    if prior_p is None:
        prior_p = [0.3, 0.4, 0.3]

    np.random.seed(233)

    n_sample = (np.array(prior_p)*n).astype(int)

    s0 = np.random.multivariate_normal(mu[0], sigma, n_sample[0])
    s1 = np.random.multivariate_normal(mu[1], sigma, n_sample[1])
    s2 = np.random.multivariate_normal(mu[2], sigma, n_sample[2])

    l0 = np.array([0] * n_sample[0]).reshape(1,-1)
    l1 = np.array([1] * n_sample[1]).reshape(1,-1)
    l2 = np.array([2] * n_sample[2]).reshape(1,-1)

    sample = np.concatenate((s0,s1,s2), axis=0)
    label = np.concatenate((l0,l1,l2), axis=1)
    data = np.insert(sample, sample.shape[1], values=label, axis=1)
    # print(data.shape)
    np.random.shuffle(data)
    # print(data)

    # sum = 0
    # for i in range(1000):
    #     sum += data[i][2]
    # print(sum)

    # plt.plot(s0[:,0], s0[:,1], 'o')
    # plt.plot(s1[:,0], s1[:,1], 'o')
    # plt.plot(s2[:,0], s2[:,1], 'o')
    # plt.show()

    with open("data.data", "wb") as f:
        np.save(f, data)

def load_data():
    '''
    load_data.

    :return:
        n: # of samples;
        sample: n by d ndarray;
        label: n by 1 ndarray.
    '''
    with open("data.data", "rb") as f:
        data2 = np.load(f)
    n = data2.shape[0]
    sample = data2[:,:-1]
    label = data2[:,-1:]
    label = label.astype(int)

    # print(data2.shape)
    # print(sample.shape)
    # print(label)

    return n, sample, label

def split_data(n, sample, label):
    percentage = 0.7
    n1 = int(n * percentage)
    sample_train = sample[:n1]
    sample_test = sample[n1:]
    label_train = label[:n1]
    label_test = label[n1:]
    return sample_train, label_train, sample_test, label_test

def softmax(x):
    # print(x)
    return np.exp(x)/np.sum(np.exp(x))

class Discriminative_Model:
    def __init__(self, c, d):
        self.c = c
        self.d = d
        self.wt = np.zeros((self.c,self.d+1))

    def fit(self, n, x, y, alpha=1.0, epoch=500, batch=10):
        w0 = copy.deepcopy(self.wt)
        # print(w0)
        index = np.array([i for i in range(n)])
        # print(index)
        # print(x)
        b = np.ones(x.shape[0])
        x = np.c_[x, b]
        # print(x)
        i=n

        temp = (copy.deepcopy(x[0])).reshape(-1,1)
        # print(w0.shape)
        # print(w0[:,1])
        # print(temp*1.0)
        # print(temp.shape)
        #
        # print(softmax(np.dot(w0,temp)).shape)
        # sum = 0
        # dec = alpha / epoch

        for cur in range(epoch):
            if i==n:
                i = 0
                np.random.shuffle(index)
            j = min (i+batch, n)
            dw = np.zeros((3,3))
            current_batch = j - i
            for k in range(i,j):
                # if i ==0 and sum == 1:
                #     print(index[k])
            # sum+=1
                temp = (copy.deepcopy(x[index[k]])).reshape(-1,1)
                temp2 = softmax(np.dot(w0, temp))

                # assert(temp2.shape[0] == 3)

                for QWQ in range(3):
                    if y[index[k]] == QWQ:
                        flag = 1.0
                    else :
                        flag = 0.0
                    temp3 = temp * (flag - temp2[QWQ][0])
                    for QEQ in range(3):
                        dw[QEQ][QWQ] += temp3[QEQ][0]
                        # dw[:, QWQ] += temp*(flag-temp2[QWQ][0])
            w0=self.wt
            self.wt += dw.T/current_batch * alpha
            # alpha -= dec
        return self.wt

    def pred(self, w, x, real_y):
        # print(x.shape[0])
        b = np.ones(x.shape[0])
        x = np.c_[x, b]
        ac = 0
        for i in range(x.shape[0]):
            temp = (copy.deepcopy(x[i])).reshape(-1, 1)
            temp2 = softmax(np.dot(w, temp))
            mx = 0
            for j in range(temp2.shape[0]):
                if temp2[j][0] > temp2[mx][0]:
                    mx = j
            if real_y[i] == mx:
                ac += 1
        acc = ac*1.0/x.shape[0]
        print("Discriminative Model Accuracy:")
        print(acc)
        return

class Generative_Model:
    def __init__(self, c, d):
        self.c = c
        self.d = d
    def fit(self, n, x, y):
        p = np.zeros((self.c))
        mu = np.zeros((self.c, self.d))
        sigma = np.zeros((self.d, self.d))
        for i in range(n):
            mu[y[i]] += x[i]
            p[y[i]] += 1
        for i in range(self.c):
            mu[i] /= p[i]
        for i in range(self.c):
            p[i] /= n

        for i in range(n):
            temp = x[i] - mu[y[i]]
            sigma += np.dot(temp.reshape(-1,1), temp.reshape(1,-1))
        sigma /= n

        # print(mu)
        # print(p)
        # print(sigma)
        return p, mu, sigma
    def pred(self, p, mu, sigma, x, real_y):
        # print(mu.shape)
        ac = 0
        for i in range(x.shape[0]):
            pos = np.zeros((3))
            for j in range(self.c):
                var = multivariate_normal(mean=mu[j], cov=sigma)
                likelihood = var.pdf(x[i])
                pos[j] = likelihood * p[j]
            pos = softmax(pos)
            # if i==0:
            #     print(pos)
            mx = 0
            for j in range(self.c):
                if pos[j] > pos[mx]:
                    mx = j
            if real_y[i] == mx:
                ac += 1
        acc = ac * 1.0 / x.shape[0]
        print("Generative Model Accuracy:")
        print(acc)
        return

def run_D_M(C, sample_train, label_train, sample_test, label_test):
    model_1 = Discriminative_Model(C, sample.shape[1])
    alpha = 0.01
    epoch = 1000
    batch = 10
    wt = model_1.fit(sample_train.shape[0], sample_train, label_train, alpha, epoch, batch)
    # print(wt)
    model_1.pred(wt, sample_test, label_test)
    return

def run_G_M(C, sample_train, label_train, sample_test, label_test):
    model_2 = Generative_Model(C, sample.shape[1])
    p, mu, sigma = model_2.fit(sample_train.shape[0], sample_train, label_train)
    model_2.pred(p, mu, sigma, sample_test, label_test)
    return

if __name__ == '__main__':
    n = 1000
    generate_data(n)
    n, sample, label = load_data()
    sample_train, label_train, sample_test, label_test = split_data(n, sample, label)
    C = 3
    run_D_M(C, sample_train, label_train, sample_test, label_test)
    run_G_M(C, sample_train, label_train, sample_test, label_test)
