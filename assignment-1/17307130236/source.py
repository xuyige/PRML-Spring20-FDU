import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def gen_gaussian_clusters(ns, mus, sigma):
    """
    Generate three clusters of gaussian distributed samples with specific expectation and covariance matrix.

    Inputs:
    -ns : array_like,with shape of (3,)
        scale of each cluster
    -mus : array_like,with shape of (n,2)
        Expectations of gaussian distributions.
    -sigma : array_like,with shape of (2,2)
        Covariance matrix of gaussian distributions.

    Returns:
    -data :array,with shape of (n,2)
        samples compiled to gaussian distribution,
    -labels :array,with shape of(n,)
        Sample labels.
    """
    ns = np.asarray(ns)
    mus, sigma = np.asarray(mus), np.asarray(sigma)
    dim = mus.shape[1]
    data = np.zeros(shape=(np.sum(ns), dim))
    labels = np.zeros(shape=(np.sum(ns)))
    last = 0
    for i in range(mus.shape[0]):
        mu = mus[i]
        n = ns[i]
        data[last:last+n] = np.random.multivariate_normal(mu, sigma, n)
        labels[last:last+n] = i*np.ones(n, dtype='int32')
        last += n
    return np.round(data, 4), labels


def create_dataset(mus, sigma, ns, filename):
    """
    Create dataset based on the gaussian distributions generated and shuffle it.
    Then, write the dataset in the file to store it.

    Inputs:
    -mus : array_like
        Expectations of gaussian distributions.
    -sigma : array_like
       Covariance matrix of gaussian distributions.
    -ns : array_like
        scale of each cluster

    """
    data, labels = gen_gaussian_clusters(ns, mus, sigma)
    dataset = ''
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            dataset = dataset + '{:.4f} '.format(data[i, j])
        dataset = dataset + str(int(labels[i])) + '\n'

    f = open(filename, 'w')
    f.write(dataset)
    f.close()


def show_scatter(data, ns):
    """
    Visualize the distribution of 3 clusters.
    """
    last = 0
    c = ['r', 'g', 'b']
    for i in range(len(ns)):
        x, y = data[last:last+ns[i]].T
        plt.scatter(x, y, marker='o', color=c[i])
        last = last+ns[i]
    plt.axis()
    plt.title("3-cluster gaussian distributed samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def load_dataset(filename):
    """
    Write the dataset in the file to store it.

    Returns:
    -data :array,with shape of (n,2)
        samples compiled to gaussian distribution,
    -labels :array,with shape of(n,)
        Sample labels.
    """

    data = []
    labels = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            data.append([i for i in line.split(' ')][:-1])
            labels.append([i for i in line.split(' ')][-1])
    return np.asarray(data, dtype='float32'), np.array(labels, dtype='int32')


def one_hot_vector(number, n_classes):
    """
    Get one-hot vector for a specific class.

    Inputs:
    -number :class id
    -n_classes : number of classes
    Returns:
    -out :one-hot vector for the class

    """

    out = np.zeros([n_classes])
    out[number] = 1
    return out


def get_one_hot(y, n_classes):
    """
    Get one-hot vector for a group of labels.

    Inputs:
    -y :a group of labels
    -n_classes : number of classes
    Returns:
    -out :one-hot vector for a group of labels

    """
    out = []
    for i in range(y.shape[0]):
        out.append(one_hot_vector(y[i], n_classes))

    out = np.array(out, dtype='int32')
    return out

class DiscriminativeClassifier:
    def __init__(self):
        self.theta = None
        self.cost = []
        self.acc = []
    
    def softmax(self,X):
        """
        Softmax function applying for Multiclass Classification.
        Inputs:
        -X:A numpy array of samples.
        Returns:
        -A numpy array of samples after applying softmax function, value in range 0 to 1.
        """
        scores = X.dot(self.theta)
        shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
        return np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1,1)
     
    def cal_cost_and_grad(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative. 
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of the whole samples; each sample has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        Returns: 
        -loss: cost on this minibatich
        -dW: gradient with respect to self.theta
        
        """
        loss = 0.0
        n,dim = X_batch.shape
        k = np.max(y_batch) + 1
        dW = np.zeros_like(self.theta)
        softmax_output = self.softmax(X_batch)
        loss = -np.sum(np.log(softmax_output[range(n), list(y_batch)]))
        loss /= n
        y_actual = get_one_hot(y_batch,k).reshape(n,k)
        dW = np.dot(X_batch.T,(y_actual-softmax_output.reshape(n,k)))
        dW /= n
        return loss,dW
    
    def training(self, dataset, labels, lr, batch_size, EPOCH):
        """
        Train this linear discriminative classifier using mini-batch gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - lr: (float) learning rate for optimization.
        - batch_size: (integer) number of training examples to use at each step.
        - EPOCH: times for training the whole dataset once.
        
        """
        n, dim = dataset.shape
        k = int(labels.max()+1)
        if self.theta is None:
            self.theta = 0.001 * np.random.randn(dim, k)
        print('Epoch \t\t Acc')
        for epoch in range(EPOCH):
            data_and_lab = list(zip(dataset,labels))
            np.random.shuffle(data_and_lab)
            dataset, labels = zip(*data_and_lab)
            dataset = np.array(dataset)
            labels = np.array(labels)
            for i in range(n //batch_size + 1):
                begin = i * batch_size
                end = (i + 1) * batch_size
                if begin >= n:
                    continue
                if end > n:
                    end = n
                mini_dataset = dataset[begin:end]
                mini_labels = labels[begin:end]
                loss,grad = self.cal_cost_and_grad(mini_dataset,mini_labels)
                self.theta += lr * grad
                self.cost.append(loss)
            pred = self.predict(dataset)
            accuracy=np.mean(np.equal(get_one_hot(pred,k), get_one_hot(labels,k)))
            self.acc.append(accuracy)
            
            print('{epoch}\t\t{acc}%'.format_map({"epoch": epoch + 1, "acc": accuracy*100}))           
        self.plot_metrics()

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.
        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        scores = self.softmax(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred  
    
    def plot_metrics(self):
        """
        Visualize the trend of loss.
        """
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.title('Log-likelyhood')
        plt.show()
        plt.plot(np.arange(len(self.acc)), self.acc)
        plt.title('Accuracy')
        plt.show()
class GenerativeClassifier:
    def __init__(self):
        self.ps = None
        self.mus = None
        self.sigma = None

    def training(self,data,labels):
        """
        Train this linear generative classifier using parametric density estimation.
        Inputs:
        - data: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - labels: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        """
        n,dim=data.shape
        k = np.max(labels) + 1
        self.ps = np.zeros(k)
        self.mus = np.zeros((k,dim))
        self.sigma = np.zeros((dim,dim))
        for i in range(n):
            self.ps[labels[i]] += 1
            self.mus[labels[i]] += data[i]
        for i in range(k):
            self.mus[i] /= self.ps[i]
            self.ps[i] /=n    
        for i in range(n):
            self.sigma += np.dot(np.asmatrix(data[i]-self.mus[labels[i]]).T,np.asmatrix(data[i]-self.mus[labels[i]]))
        self.sigma /= n
        
    def predict(self,data,labels):
        """
        Use the estimated parametric density of this linear classifier to predict labels for
        data points.
        Inputs:
        - data: N x D array of training data. Each column is a D-dimensional point.
        Returns:
        - acc: The accuracy on the test dataset.
        """
        n = data.shape[0]
        k = np.max(labels)+1
        acc= 0.0
        prob = np.zeros(k)
        col = ['ro','go','bo']
        wrong = ['rx','gx','bx']
        plt.figure(figsize=(10,10))
        for i in range(n):
            for j in range(k):
                prob[j] = stats.multivariate_normal.pdf(data[i],self.mus[j],self.sigma)*self.ps[j]
            y_pred = np.argmax(prob)
            if y_pred==labels[i]:
                plt.plot ( data[i][0] , data[i][1], col[labels[i]] )
                acc+=1
            else:
                plt.plot ( data[i][0] , data[i][1], wrong[labels[i]] )
        acc /= n
        plt.show ()
        return acc
           
mus=[[0,2],[-2,0],[2,0]]
sigma=[[.2,0],[0,.2]]
ns=[1000,1000,1000]
filename='my.data'
create_dataset(mus,sigma,ns,filename)
data,labels=load_dataset(filename)
show_scatter(data,ns)

data_and_lab = list(zip(data, labels))
np.random.shuffle(data_and_lab)
data, labels = zip(*data_and_lab)
data = np.array(data)
labels = np.array(labels)
k = int(max(labels))+1
train_data,train_labels=data[:2000],labels[:2000]
test_data,test_labels=data[2000:],labels[2000:]

print("Discriminative Classifier:")
dc=DiscriminativeClassifier()
dc.training(train_data,train_labels,lr=0.1,batch_size=128,EPOCH=10)
predict = dc.predict(test_data)
accuracy=np.mean(np.equal(get_one_hot(predict,k), get_one_hot(test_labels,k)))
print('Accuracy is {acc}%'.format_map({"acc": accuracy*100}))

print("Generative Classifier:")
gc = GenerativeClassifier()
gc.training(train_data,train_labels)
acc = gc.predict(test_data,test_labels)
print('Accuracy is {acc}%'.format_map({"acc": acc*100}))