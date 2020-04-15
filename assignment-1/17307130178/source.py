import numpy as np
import matplotlib.pyplot as plt
import random

# Part I
# In this part, you are going to design 3 gaussian distribution, each of distribution indicates label A, 
# label B, and label C, respectively. Then construct a dataset sampling from these distributions.

mean = np.array([[1,0],[5,2],[2,5]])
cov = np.array([[1,0],[0,1]])

class gaussian():
    mean = None
    cov = None 
    sample = None
    G = None
    label = ''
    def __init__(self, mean, cov, sample, label):
        self.mean = mean
        self.cov = cov
        self.sample = sample
        self.label = label
        self.G = np.random.multivariate_normal(mean,cov,sample)
        self.show()
    def show(self):
        plt.plot(*self.G.T, '.', label = self.label)
        plt.axis('scaled')
        plt.legend()
        # plt.show()
    def dataset(self):
        x = self.G.T[0]
        y = self.G.T[1]
        return [[x[i], y[i], self.label] for i in range(len(x))]


def gaussian_distribution(tofile=False, showplt=False, sample=500):
    label = ['A','B','C']
    gaussian_A = gaussian(mean[0],cov,sample,label[0])
    gaussian_B = gaussian(mean[1],cov,sample,label[1])
    gaussian_C = gaussian(mean[2],cov,sample,label[2])
    if showplt:
        plt.show()
    dataset = gaussian_A.dataset() + gaussian_B.dataset() + gaussian_C.dataset()
    random.shuffle(dataset)
    if tofile:
        with open('dataset.data','w') as f:
            print(dataset,file=f)
    else:
        return dataset
    print("下面是数据样例：")
    print(dataset[:5])


def data_preprocessing(dataset):
    random.shuffle(dataset)
    size = len(dataset)
    test_data, training_data = dataset[:size//5], dataset[size//5:]
    return test_data, training_data


# Part II
# In this part, you are required to construct 2 linear classification models: a generative model and 
# a discriminative model. Meanwhile, you are required to compare their differences.

# Generative models:
# 1. Use maximum likelihood to find the root
# 2. Can be done without SGD
class generative_model():
    def __init__(self):
        self.W = None
        self.W0 = None
        self.label = ['A','B','C']

    def train(self, training_data):
        label = self.label
        data_A = [[x,y] for [x,y,l] in training_data if l == label[0]]
        data_B = [[x,y] for [x,y,l] in training_data if l == label[1]]
        data_C = [[x,y] for [x,y,l] in training_data if l == label[2]]
        data = [data_A, data_B, data_C]
        # Prior
        Pi = len(data_A)/len(training_data)
        # mean
        u = [np.sum(data[i], axis=0).reshape((2,1))/len(data[i]) for i in range(3)]
        # Sigma
        Sigma = np.zeros((2,2))
        for i in range(3):
            for k in range(len(data[i])):
                x = np.array(data[i][k]).reshape((2,1))
                # u[i] = u[i].reshape((2,1))
                Sigma += np.dot(x-u[i], (x-u[i]).T)
        Sigma /= len(training_data)
        
        Sigma = np.matrix(Sigma)
        self.W = [np.dot(Sigma.I,u[i]) for i in range(3)]
        self.W0 = [-0.5*u[i].T.dot(self.W[i]) + np.log(Pi) for i in range(3)]
    def softmax(self, a):
        a = np.exp(a)
        return a / np.sum(a)
    def predict(self, test_data):
        label = self.label
        predict_Y = []
        Y = [x[2] for x in test_data]
        num_correct, num_false = 0,0
        for X in test_data:
            x = np.array(X[:2]).reshape((2,1))
            y = X[2]
            a = [ np.dot(self.W[i].T, x) + self.W0[i] for i in range(3) ] 
            predict_y = label[np.argmax(self.softmax(a))]
            # print(predict_y,y)
            predict_Y.append(predict_y)
            num_correct += (predict_y==y)
        accuracy = num_correct / len(Y)
        print("Generative model, accuracy: ", accuracy)
        return predict_Y
        

class discriminative_model(object):
    def __init__(self):
        self.W = None
        self.W0 = None

    def train(self, training_data, learning_rate=1e-3, reg=1e-5, num_iters=2000,
              batch_size=200, verbose=False):
        # X(N,2) y(N,)
        X = np.array([[x[0],x[1]] for x in training_data])
        y = np.array([ord(x[2])-ord('A') for x in training_data])
        num_train, dim = X.shape
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)
        if self.W0 is None:
            self.W0 = 0 * np.random.randn(num_classes)
        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            
            index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[index, :]
            y_batch = y[index]
            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            # perform parameter update
            self.W += learning_rate*grad['W']
            self.W0 += learning_rate*grad['W0']
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history

    def predict(self, test_data):
        X = np.array([[x[0],x[1]] for x in test_data])
        y = np.array([ord(x[2])-ord('A') for x in test_data])
        y_pred = np.zeros(X.shape[0])  # 1 x N
        # X(N, D) W(D, C) y_pred(1,N)
        scores = np.dot(X, self.W)   # (N, C)
        scores += self.W0
        scores = 1 / (1+np.exp(-scores))
        y_pred = np.argmax(scores, axis=1)
        num_correct = np.sum(y==y_pred)
        print("Discriminative model, accuracy: ",num_correct/len(y))
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        loss = 0.0
        dW = np.zeros_like(self.W)
        dW0 = np.zeros_like(self.W0)
        grad = {}
        num_train = X_batch.shape[0]
        num_classes = self.W.shape[1]
        scores = np.dot(X_batch,self.W) + self.W0
        scores = 1/(1+np.exp(-scores))
        margin = np.zeros_like(scores)
        margin[np.arange(num_train),y_batch] += 1
        loss = 0.5 * np.sum(np.square(scores-margin)) + 0.5*reg*np.sum(self.W*self.W)
        dW = np.dot(X_batch.T, margin-scores)
        dW0 = np.sum(margin-scores, axis=0).reshape((num_classes,))
        grad['W'] = dW
        grad['W0'] = dW0    
        return loss, grad



# Part III
# In this part, you can reorganize the scale of your dataset or the adjust the overlap between 
# different gaussian distributions.


def test(m=mean):
    global mean
    mean = m
    print(m)
    dataset = gaussian_distribution(showplt=True,sample=500)
    test_data, training_data = data_preprocessing(dataset)
    model = generative_model()
    model.train(training_data)
    model.predict(test_data)
    model = discriminative_model()
    model.train(training_data)
    model.predict(test_data)

def test_sample():
    samples = [10,50,100,500,1000]
    for i in range(5):
        dataset = gaussian_distribution(showplt=True,sample=samples[i])
        print("Sample points:",samples[i])
        test_data, training_data = data_preprocessing(dataset)
        model = generative_model()
        model.train(training_data)
        model.predict(test_data)
        model = discriminative_model()
        model.train(training_data)
        model.predict(test_data)

def test_overlapping():
    global mean
    mean = np.array([[1,0],[1,1],[1,2]])
    test(mean)
    mean = np.array([[1,0],[2,3],[3,2]])
    test(mean)
    mean = np.array([[1,0],[4,2],[2,4]])
    test(mean)
    mean = np.array([[1,0],[5,2],[2,5]])
    test(mean)
    mean = np.array([[1,0],[6,2],[2,6]])
    test(mean)



if __name__ == "__main__":
    test(mean)

    # IF YOU NEED TO TEST,RUN CODES BELOW
    # test_sample()
    # test_overlapping()
