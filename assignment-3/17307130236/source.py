import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random
import math

# data
def gen_sample(mu, cov, n):
    return np.random.multivariate_normal(mean=mu, cov=cov, size=n)

def gen_dataset(k, mus, covs, ns, filename):
    f = open(filename, 'w')
    dataset = ''
    for i in range(k):
        mu = mus[i]
        cov = covs[i]
        n = ns[i]
        data = gen_sample(mu, cov, n)
        for j in range(data.shape[0]):
            for k in range(data.shape[1]):
                dataset = dataset + '{:.4f} '.format(data[j, k])
            dataset = dataset + str(i) + '\n'

    f.write(dataset)
    f.close()


def load_dataset(filename):
    data = []
    labels = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            data.append([i for i in line.split(' ')][:-1])
            labels.append([i for i in line.split(' ')][-1])
    return np.asarray(data, dtype='float32'), np.array(labels, dtype='int32')


def scale_data(Y):
    for i in range(0, 1):
        maxval = Y[:, i].max()
        minval = Y[:, i].min()
        Y[:, i] = (Y[:, i] - minval) / (maxval - minval)
    return Y


def show_scatter(data, ns):
    last = 0
    c = ['r', 'g', 'b']
    for i in range(len(ns)):
        x, y = data[last:last+ns[i]].T
        plt.scatter(x, y, marker='o', color=c[i])
        last = last+ns[i]
    plt.axis()
    plt.title("gaussian samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# model
class K_Means_Plus_Plus():
    def __init__(self, points_list, k):
        self.center_count = 0
        self.points_list = list(points_list)
        self.cluster_count = k
        self.initialize_first_center()
        self.initialize_other_centers()

    def initialize_first_center(self):
        self.center_list = []
        index = random.randint(0, len(self.points_list)-1)

        self.center_list.append(self.remove_point(index))
        self.center_count = 1

    def remove_point(self, index):
        new_center = self.points_list[index]
        del self.points_list[index]
        return new_center


    def initialize_other_centers(self):
        while not self.is_finished():
            distances = self.find_smallest_distances()
            chosen_index = self.choose_weighted(distances)
            self.center_list.append(self.remove_point(chosen_index))
            self.center_count += 1

    def find_smallest_distances(self):
        distance_list = []
        for point in self.points_list:
            distance_list.append(self.find_nearest_center(point))

        return distance_list

    def find_nearest_center(self, point):
        min_distance = math.inf
        for values in self.center_list:
            distance = self.euclidean_distance(values, point)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def choose_weighted(self, distance_list):
        distance_list = [x**2 for x in distance_list]
        weighted_list = self.get_weight(distance_list)
        indices = [i for i in range(len(distance_list))]
        return np.random.choice(indices, p = weighted_list)

    def get_weight(self, list):
        sum = np.sum(list)
        return [x/sum for x in list]

    def euclidean_distance(self, point1, point2):
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
        return np.linalg.norm(point2-point1)

    def is_finished(self):
        if self.center_count == self.cluster_count:
            return True
        else:
            return False    
    
def phi(Y, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)

class GMM():
    def __init__(self,n_clusters):
        self.mu = None
        self.cov = None
        self.alpha = None
        self.k = n_clusters
        self.loss = []
        
    def init_params(self, shape):
        N, D = shape
        self.mu = np.random.rand(self.k, D)
        self.cov = np.array([np.eye(D)] * self.k)
        self.alpha = np.array([1.0 / self.k] * self.k)
    
    def getExpectation(self, x):
        N = x.shape[0]
        K = self.alpha.shape[0]
        gamma = np.mat(np.zeros((N, K)))
        prob = np.zeros((N, K))
        for k in range(K):
            prob[:, k] = phi(x, self.mu[k], self.cov[k])
        prob = np.mat(prob)
        for k in range(K):
            gamma[:, k] = self.alpha[k] * prob[:, k]
        for i in range(N):
            gamma[i, :] /= np.sum(gamma[i, :])
        return gamma
    
    def maximize(self, x, gamma):
        N, D = x.shape
        K = gamma.shape[1]
        mu = np.zeros((K, D))
        cov = []
        alpha = np.zeros(K)
        for k in range(K):
            Nk = np.sum(gamma[:, k])
            mu[k, :] = np.sum(np.multiply(x, gamma[:, k]), axis=0) / Nk
            cov_k = (x - mu[k]).T * np.multiply((x - mu[k]), gamma[:, k]) / Nk
            cov.append(cov_k)
            alpha[k] = Nk / N
        cov = np.array(cov)
        self.mu = mu
        self.cov = cov
        self.alpha = alpha
    
    def train(self, x, iteration):
        N = x.shape[0]
        for j in range(int(iteration/4)+1):
            fig=plt.figure()
            res = iteration-j*4
            for i in range(min(4,res)):
                gamma = self.getExpectation(x)
                self.maximize(x, gamma)
                clusters = self.predict(x)
                colors = ['r','g','b']
                ax=fig.add_subplot(2,2,i+1)
                for i in range(len(clusters)):
                    cluster = clusters[i]
                    if cluster.shape[0]>0:
                        ax.scatter(cluster[:, 0], cluster[:, 1], marker='o',color=colors[i%3],s=20)
                ax.scatter(self.mu[:, 0], self.mu[:, 1], marker='x',color='black',s=60)
                P = np.zeros((N,self.k))
                for k in range(self.k):
                    P[:,k] = phi(x, self.mu[k], self.cov[k])

                self.loss.append(-np.sum(np.log(P.dot(self.alpha)))/N)
            plt.show()       
    
    def predict(self, x):
        N = x.shape[0]
        gamma = self.getExpectation(x)
        category = gamma.argmax(axis=1).flatten().tolist()[0]
        clusters = []
        for j in range(self.k):
            clusters.append(np.array([x[i] for i in range(N) if category[i] == j]))
        return clusters
    
    def plot_metrics(self):
        plt.plot(np.arange(len(self.loss)), self.loss)
        plt.title('Log-likelyhood')
        plt.show()
        
# test
cov1 = np.mat("0.3 0;0 0.1")
cov2 = np.mat("0.2 0;0 0.3")
cov3 = np.mat("0.1 0;0 0.1")
mu1 = np.array([10, 1])
mu2 = np.array([8, 0])
mu3 = np.array([8, 1])
mus = [mu1, mu2, mu3]
covs = [cov1, cov2, cov3]
ns = [30, 70, 60]
filename = 'my.data'

gen_dataset(3,mus,covs,ns,filename)
data, labels = load_dataset(filename)
# data = scale_data(data)
show_scatter(data, ns)
data_and_lab = list(zip(data, labels))
np.random.shuffle(data_and_lab)
data, labels = zip(*data_and_lab)
data = np.asarray(data)
labels = np.asarray(labels)

# random initialization
model = GMM(3)
model.init_params(data.shape)
model.train(data, 60)
model.plot_metrics()


# k-means initialization
model1 = GMM(3)
test = K_Means_Plus_Plus(data, 3)
mus = test.center_list
model1.init_params(data.shape)
model1.mu = mus
model1.train(data, 60)
model1.plot_metrics()
