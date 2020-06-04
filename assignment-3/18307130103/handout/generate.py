import copy
import numpy as np
import matplotlib.pyplot as plt

def Outputdata(sample, label, filename='1'):
    n = sample.shape[0]
    data = ''
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            data += str(sample[i][j]) + ' '
        data += str(label[i]) + '\n'
        
    with open('./' + filename + '.data', 'w') as f:
        f.write(data)
        

def Generate(mean = [[0, 0], [-4, -4], [3, 5]], cov = [[[1, 0], [0, 1]], [[1.5, 0], [0, 1.5]], [[2, 1], [1, 2]]], prior = [.3, .3, .4], n = 500, filename = "1"):
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
            ./filename.data
        return:
            sample, label
            two arrays indicate the points of the distribution and the label of these points
    '''
    m = len(mean)
    tot = [int (n * prior[i]) for i in range(m)]
    sample = np.random.multivariate_normal(mean[0], cov[0], tot[0])
    label = np.asarray([0 for j in range(tot[0])])
    for i in range(m - 1):
        sample_ = np.random.multivariate_normal(mean[i + 1], cov[i + 1], tot[i + 1])
        label_ = np.asarray([i + 1 for j in range(tot[i + 1])])
        sample = np.append(sample, sample_, 0)
        label = np.append(label, label_, 0)
    
    label_ = copy.deepcopy(label)
    sample_ = copy.deepcopy(sample)
    p = np.random.permutation(label.shape[0])
    for i in range(sample_.shape[0]):
        sample[i] = sample_[p[i]]
        label[i] = label_[p[i]]

    data = ''
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            data += str(sample[i][j]) + ' '
        data += str(label[i]) + '\n'
    
    with open('./' + filename + '.data', 'w') as f:
        f.write(data)

def Inputdata(filename = '1'):
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

def Picture(sample, label, filename = '1', mean = np.zeros((0, 2)), color = ['#FF0000', '#0000FF', '#008000', '#000000']):
    '''
        get the picture of 2D data, and save the picture in './filename.png'
        paremeter sample:
            the coordinate of points in the data
        paremeter sample:
            the label of points in the data
    '''
    n = sample.shape[0]
    m = max(label) + 1
    x = []
    y = []
    for i in range(m + 1):
        x.append([])
        y.append([])
    for i in range(n):
        x[label[i]].append(sample[i][0])
        y[label[i]].append(sample[i][1])
    n = mean.shape[0]
    for i in range(n):
        x[m].append(mean[i][0])
        y[m].append(mean[i][1])
    for i in range(m + 1):
        plt.scatter(x[i], y[i], s = 10, c = color[i])
    filename = './' + filename + '.png'
    plt.savefig(filename, dpi = 300)
    plt.cla()

if __name__ == "__main__":
    Generate()
    sample, label = Inputdata()
    Picture(sample, label)