import math
import copy
import numpy as np
import matplotlib.pyplot as plt

def createdata(mean = [[0, 0], [-2, -2], [3, -2]], cov = [[1, 0], [0, 1]], prior = [.3, .3, .4], n = 500, filename = '*'):
    '''
        create data with Gaussian distribution.
        each of point labeld 0, 1, 2 respectively
        parameter mean:
            a 3*3 list,
            mean[0], mean[1], mean[2] indicates the vector miu of the first, second, third distribution
        parameter cov:
            a 3*3 list indicates the covariance matrix of the three distributions
        parameter prior:
            a list indicates the prior probability of the three distributions
        parameter n:
            a int indicates the total number of points in three distributions
        parameter filename:
            the name of the data file you want to output. 
        output file:
            ./_.data
        return:
            sample, label
            two arrays indicate the points of the distribution and the label of these points
    '''
    tot = [int(n * prior[i]) for i in range(3)]
    sample = np.random.multivariate_normal(mean[0], cov, tot[0])
    sample = np.append(sample, np.random.multivariate_normal(mean[1], cov, tot[1]), 0)
    sample = np.append(sample, np.random.multivariate_normal(mean[2], cov, tot[2]), 0)
    label = np.asarray([0 for i in range(tot[0])], dtype = int)
    label = np.append(label, np.asarray([1 for i in range(tot[1])], dtype = int), 0)
    label = np.append(label, np.asarray([2 for i in range(tot[2])], dtype = int), 0)
    N = sample.shape[0]
    p = np.random.permutation(N)
    sample_ = copy.deepcopy(sample)
    label_ = copy.deepcopy(label)
    for i in range(N):
        sample[i, :] = sample_[p[i], :]
        label[i] = label_[p[i]]
    data = ''
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            data += str(sample[i][j]) + ' '
        data += str(label[i]) + '\n'
        
    with open('./' + filename + '.data', 'w') as f:
        f.write(data)


def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))

def argmax(x):
    mx = x[0][0]
    am = 0
    for i in range(x.shape[1]):
        if mx < x[0][i]:
            mx = x[0][i]
            am = i
    return(am)

class DiscriminativeModel:

    def __init__(self):
        self.n = 0
        self.w = np.zeros([0,])
        self.w0 = np.zeros([0,])

    def build(self, sample, label, batch = 128, epoch = 100):
        '''
            get the parameters of the discriminative model from the given data
            parameter sample:
                the points of the train set
            parameger label:
                the label of the points 
            parameter batch:
                the size of mini_batch
            parameter epoch:
                the number of epochs
        '''
        self.n = label.max() + 1
        n = sample.shape[0]
        d = sample.shape[1]
        t = np.zeros([n, self.n])
        for i in range(n):
            t[i][label[i]] = 1
        self.w = np.zeros([self.n, d])
        self.w0 = np.zeros([1, self.n])
        batch_num = n // batch
        if (n % batch != 0):
            batch_num += 1
        p = [i for i in range(n)]
        for i in range(epoch):
            p = np.random.permutation(n)
            for j in range(batch_num):
                bg = j * batch
                ed = (j + 1) * batch
                if (ed > n):
                    ed = n
                x = np.zeros([ed - bg, d])
                for k in range(ed - bg):
                    x[k: k + 1] = sample[p[k + bg]: p[k + bg] + 1]
                y = np.dot(x, self.w.T)
                for k in range(ed - bg):
                    y[k: k + 1] = y[k: k + 1] + self.w0
                    y_ = softmax(y[k: k + 1])
                    delta = (t[p[k + bg]: p[k + bg] + 1] - y_)
                    self.w += np.dot(delta.T , x[k: k + 1])
                    self.w0 += delta


    def query(self, sample):
        '''
            get the classification result of the point array
            parameter sample:
                an array indicates the points to classify
            return:
                a array rs indicates the label of the input points
        '''
        y = self.w0
        rs = np.zeros(sample.shape[0], dtype = int)
        for i in range(sample.shape[0]):
            x = sample[i: i+1]
            y = np.dot(x, self.w.T) + self.w0
            t = softmax(y)
            rs[i] = argmax(t)
        return(rs)

  
class GenerativeModule:

    def __init__(self):
        self.n = 0
        self.w = np.zeros([0,])
        self.w0 = np.zeros([0,])
    
    def build(self, sample, label):
        '''
            get the parameters of the generative model from the given data
            parameter sample:
                the points of the train set
            parameger label:
                the label of the points 
            parameter batch:
                the size of mini_batch
            parameter epoch:
                the number of epochs
        '''
        self.n = label.max() + 1
        d = sample.shape[1]
        n = sample.shape[0]
        miu = np.zeros([self.n, d])
        sigma = np.zeros([d, d])
        classnum = np.zeros([self.n])
        self.w = np.zeros([self.n, d])
        self.w0 = np.zeros([1, self.n])
        for i in range(n):
            classnum[label[i]] += 1
        for i in range(n):
            miu[label[i]: label[i] + 1] += sample[i: i + 1] / classnum[label[i]]
        for i in range(n):
            vc = sample[i: i+1] - miu[label[i]: label[i] + 1]
            sigma = sigma + np.dot(vc.T, vc) / n
        sigma_inv = np.asmatrix(sigma).I
        for i in range(self.n):
            self.w[i : i + 1] += np.asmatrix(miu[i : i + 1]) * sigma_inv.T
            self.w0[0][i] += -1 / 2 * np.asmatrix(miu[i : i + 1]) * sigma_inv * np.asmatrix(miu[i : i + 1]).T + math.log(classnum[i] / n)
    
    def query(self, sample):
        '''
            get the classification result of the point array
            parameter sample:
                an array indicates the points to classify
            return:
                a array rs indicates the label of the input points
        '''
        y = self.w0
        rs = np.zeros(sample.shape[0], dtype = int)
        for i in range(sample.shape[0]):
            x = sample[i: i+1]
            y = np.dot(x, self.w.T) + self.w0
            t = softmax(y)
            rs[i] = argmax(t)
        return(rs)


def inputdata(filename = '*'):
    '''
        input the data from ./filename.data
        the format of the data is x y l
        where x y indicate the coordinate of the point and l indicates the label of the point
        parameter filename:
            the name of the data file you want to read from
        return:
            sample, label
            sample indicates the coordinate of the points
            label indicates the label of the points
    '''
    with open('./' + filename + '.data', 'r') as f:
        data = f.readlines()
    n = len(data)
    sample = np.array([])
    label = np.zeros(n, dtype = int)
    for i in range(n):
        rd = list(map(str, data[i].split()))
        d = len(rd)
        point = [[float(rd[i]) for i in range(d - 1)]]
        if (sample.shape[0] == 0):
            sample = np.asarray(point)
        else:
            sample = np.append(sample, np.asarray(point), 0)
        label[i] = int(rd[d - 1])
    return sample, label

def splitdata(sample, label, rate = 0.8):
    '''
        get a train set in the dataset of the study rate 
        parameter sample:
            an array indicates the points to be splited
        paremeter sample:
            an int array indicates the labels to be splited
        parameter sample:
            a float indicates the study rate
        return:
            sample, label
            sample indicates the coordinate of the points in train date
            label indicates the label of the points in train date
    '''
    n = sample.shape[0]
    n_ = int(n * rate)
    st = np.random.randint(0, n - n_ + 1)
    sample = copy.deepcopy(sample[st: st + n_])
    label = copy.deepcopy(label[st: st + n_])
    return sample, label

def checker(y, t):
    '''
        compare the array y and t and calculate the accuracy rate of the y
        parameter y:
            the classified result of the labels
        parameter t:
            the groundtruth of the labels
        return:
            a float indicates the accuracy rate of the classified result
    '''
    n = y.shape[0]
    ac = 0
    for i in range(n):
        if (y[i] == t[i]):
            ac += 1
    return ac / n

def picture(sample, label, filename = '*'):
    '''
        get the picture of 2D data, and save the picture in './filename.png'
        paremeter sample:
            the coordinate of points in the data
        paremeter sample:
            the label of points in the data
    '''
    x = [[], [], []]
    y = [[], [], []]
    n = sample.shape[0]
    for i in range(n):
        x[label[i]].append(sample[i][0])
        y[label[i]].append(sample[i][1])
    plt.scatter(x[0], y[0], s = 10, c = '#FF0000', label = 'Class A')
    plt.scatter(x[1], y[1], s = 10, c = '#0000FF', label = 'Class B')
    plt.scatter(x[2], y[2], s = 10, c = '#008000', label = 'Class C')
    filename = './' + filename + '.png'
    plt.savefig(filename, dpi = 300)


if __name__ == "__main__":        
    createdata([[0, 0], [-2, -2], [3, -2]], [[1, 0], [0, 1]], prior = [.3, .3, .4], n = 500, filename = 'data')
    sample, label = inputdata('data')
    sample_, label_ = splitdata(sample, label, rate = 0.7)
    picture(sample, label, 'data')

    generativemodule = GenerativeModule()
    generativemodule.build(sample_, label_)
    label__ = generativemodule.query(sample)
    ac = checker(label__, label)
    picture(sample, label__, 'gen')
    print(ac)

    discriminativemodel = DiscriminativeModel()
    discriminativemodel.build(sample_, label_, batch = 2, epoch = 50)
    label__ = discriminativemodel.query(sample)
    ac = checker(label__, label)
    picture(sample, label__, 'dis')
    print(ac)