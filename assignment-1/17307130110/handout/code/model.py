import numpy
import random

class LinearGenerativeModel:
    '''
    Linear generative model for classification, using normal distributions and argmax classification strategy
    '''
    def __init__(self):
        self.K = 0          # the number of labels
        self.N = 0          # the dimension of sample point (vector x)
        self.w = None       # K * N matrix, weighted matrix
        self.b = None       # K * 1 matrix, bias matrix
        self.mean = None    # K * N matrix, the mean for the normal distribution of each label
        self.cov = None     # K * N matrix, the shared covariance matrix
        self.p_Ck = None    # a list (length == K), priori probability, p(Ck)
    

    def train(self, samples, labels):
        ''' train the model

        Arguments:
            samples: M * N matrix, where M is the size of samples. Each row is a sample point
            labels: M * 1 matrix, indicates the corresponding label
        '''
        self.K = labels.max() + 1
        self.N = samples.shape[1]
        self.w = numpy.asmatrix(numpy.zeros([self.K, self.N], dtype=numpy.float))
        self.b = numpy.asmatrix(numpy.zeros([self.K, 1], dtype=numpy.float))
        self.mean = numpy.asmatrix(numpy.zeros([self.K, self.N], dtype=numpy.float))
        self.cov = numpy.asmatrix(numpy.zeros([self.N, self.N], dtype=numpy.float))
        self.p_Ck = [0] * self.K
        
        M = samples.shape[0]    # the size of samples

        # conduct p_Ck
        count_Ck = [0] * self.K
        for i in range(M):
            count_Ck[labels[i, 0]] += 1
        for i in range(self.K):
            self.p_Ck[i] = count_Ck[i] / M
        
        # conduct mean
        for k in range(self.K):
            for i in range(M):
                if labels[i, 0] == k:
                    self.mean[k,...] += samples[i,...]
        for k in range(self.K):
            self.mean[k,...] /= count_Ck[k]
        
        # conduct cov
        tmp_cov = numpy.asmatrix(numpy.zeros([self.N, self.N], dtype=numpy.float))
        for k in range(self.K):
            for i in range(M):
                if labels[i,0] == k:
                    tmp_cov += (samples[i,...] - self.mean[k,...]).T * (samples[i,...] - self.mean[k,...])
            tmp_cov /= count_Ck[k]
            self.cov += tmp_cov * self.p_Ck[k]
        
        # calculate w
        inverse_cov = self.cov.I
        for k in range(self.K):
            self.w[k,...] = numpy.dot(inverse_cov, self.mean[k,...].T).T
        
        # calculate b
        for k in range(self.K):
            self.b[k,0] = -1 / 2 * (self.mean[k,...] * self.cov * self.mean[k, ...].T) + numpy.log(self.p_Ck[k])
        
    
    def classify(self, samples):
        ''' classify the samples according to w and b

        Arguments:
            samples: M * N matrix, where M is the size of samples. Each row is a sample point
        
        Returns:
            labels: a list, from 0 to K-1
        '''
        a = (self.w * samples.T + self.b).T  # a is M * K matrix, where a[i][j] is wj * xi + bj
        p_given_x = self.softmax(a)
        labels = numpy.argmax(p_given_x, axis=1)
        return labels
    

    def softmax(self, a):
        ''' softmax calculation

        Arguments:
            a: M * K matrix, where a[i][j] is wj * xi + bj
        
        Returns:
            p_given_x: M * K matrix, where p[i][j] = a[i][j] / sum(i-th row)
        '''
        return numpy.exp(a) / numpy.sum(numpy.exp(a), axis=1)


class LinearDiscriminativeModel:
    '''
    Linear discriminative model for classification, using softmax regression and argmax classification strategy
    '''
    def __init__(self):
        self.K = 0          # the number of labels
        self.W = None       # K * N matrix, weighted matrix
    

    def train(self, samples, labels, alpha, max_epochs, batch_size):
        ''' train the model

        Arguments:
            samples: M * N matrix, where each row is a sample point. 
                     So M is the size of samples and N is the dimension of a sample point.
            labels: M * 1 matrix, represents label values
            alpha: learning rate
            max_epochs: the maximum epochs
            batch_size: size of each batch
        '''
        self.K = labels.max() + 1
        M = samples.shape[0]    # size of samples
        N = samples.shape[1]    # dimension of a sample point
        self.W = numpy.asmatrix(numpy.zeros([self.K, N]))   

        # conduct one-hot matrix for label values
        OneHot_Y = numpy.asmatrix(numpy.zeros([M, self.K]))    
        for i in range(M):
            k = labels[i, 0]
            OneHot_Y[i, k] = 1
        
        # use softmax regression to train
        batch_nums = int(numpy.ceil(M / batch_size))  # the number of batches in each epoch
        # print('epoch\t\taccuracy')
        for epoch in range(max_epochs):
            for batch in range(batch_nums):
                # calculate begin and end
                begin = batch_size * batch
                if batch == batch_nums - 1:
                    end = min(M, batch_size * batch_nums)
                else:
                    end = batch_size * (batch + 1)
                # split from sample and OneHot_Y
                X = samples[begin:end,...]
                Y = OneHot_Y[begin:end,...]
                XW = X * self.W.T
                cal_Y = self.softmax(XW)
                # update W
                deta = numpy.asmatrix(numpy.zeros([self.K, N]))
                for i in range(end-begin):
                    deta += (Y[i,...] - cal_Y[i,...]).T * X[i,:]
                self.W += deta * alpha / (end - begin)
            cal_labels = self.classify(samples)
            accuracy = self.cal_accuracy(labels, cal_labels)
            # print('{}\t\t{}'.format(epoch, accuracy))
    

    def classify(self, samples):
        ''' use model to classify

        Arguments:
            samples: M * N matrix, where each row is a sample point. 
        
        Returns:
            labels: M * 1 matrix
        '''
        XW = samples.dot(self.W.T)
        cal_y = self.softmax(XW)
        labels = numpy.argmax(cal_y, axis=1)
        return labels

    
    def softmax(self, a):
        ''' use softmax to calculate conditional probabilities
        
        Arguments:
            a: M * K matrix, where a[i][j] is wj * xi

        Returns:
            p_Ck: M * K matrix, where p_Ck[i, j] = P(y=j|xi)
        '''
        return numpy.exp(a) / numpy.sum(numpy.exp(a), axis=1)


    def cal_accuracy(self, a, b):
        ''' calculate the accuracy between a and b

        Arguments:
            a: M * 1 matrix, standrad labels
            b: M * 1 matrix, checking labels

        Returns:
            accuracy: the accuracy between a and b
        '''
        M = a.shape[0]
        count = 0
        for i in range(M):
            if a[i, 0] == b[i, 0]:
                count += 1
        return count / M
