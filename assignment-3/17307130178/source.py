import numpy as np
import random
from matplotlib import pyplot as plt 

# Part I
# In this part, you are required to construct a 
# clustering dataset without any labels.

mean = np.array([[1,0],[4,2],[2,5]])
cov = np.array([[1,0],[0,1]])

# 定义高斯类
class gaussian():
    mean = None
    cov = None 
    sample = None
    G = None
    def __init__(self, mean, cov, sample):
        self.mean = mean
        self.cov = cov
        self.sample = sample
        self.G = np.random.multivariate_normal(mean,cov,sample)
        # self.show()
    def show(self):
        plt.plot(*self.G.T, '.')
        plt.axis('scaled')
        # plt.show()
    def dataset(self):
        x = self.G.T[0]
        y = self.G.T[1]
        return [[x[i], y[i]] for i in range(len(x))]

# 根据高斯分布生成聚类数据
def gaussian_distribution(tofile=False, showplt=False, sample=500):
    gaussian_A = gaussian(mean[0],cov,sample)
    gaussian_B = gaussian(mean[1],cov,sample)
    gaussian_C = gaussian(mean[2],cov,sample)
    
    dataset = gaussian_A.dataset() + gaussian_B.dataset() + gaussian_C.dataset()
    random.shuffle(dataset)
    dataset = np.array(dataset)
    # print(dataset.T.shape)
    if showplt:
        # plt.show()
        plt.scatter(dataset.T[0],dataset.T[1],marker='.')
        plt.title("dataset")
        plt.show()
    if tofile:
        with open('dataset.data','w') as f:
            print(dataset,file=f)
    else:
        return dataset


# Part II

# In this part, you are required to design Gaussian 
# Mixture Models to finish the unlabeled clustering task.

class GMM():
    def __init__(self,n_clusters, iteration=50,D=2):
        self.k = n_clusters
        self.iteration = iteration
        self.mu=0
        self.sigma=0
        self.PI=0
        self.D = D

    def multi_Gaussian(self, x, mu, sigma):
        return 1 / (pow( 2*np.pi, self.D/2) * pow(np.linalg.det(sigma) , 0.5)) * np.exp(-0.5*(x-mu).T.dot(np.linalg.pinv(sigma)).dot(x-mu))
    
    def calculate_Gamma(self, x, mu, sigma, PI):
        # Gamma(num_samples,k)
        k = self.k
        D = self.D
        num_samples = x.shape[0]
        gamma = np.zeros((num_samples, k))
        P = np.zeros(k)
        for n in range(num_samples):
            # P(k,)
            for i in range(k):
                P[i] = PI[i] * self.multi_Gaussian(x[n], mu[i], sigma[i] )
            for i in range(k):
                gamma[n,i] = P[i]/np.sum(P)
        return gamma
    
    def train(self,data):
        # data(n, D)
        n = data.shape[0]
        D = data.shape[1]

        # initialization
        PI = np.ones(self.k) / self.k # 先验
        mu = np.array([data[random.randint(0,len(data))] for i in range(self.k)]) # 随机选择几个值作为中心点
        sigma = np.full((self.k, D, D), np.diag(np.full(D,0.1)))

        for i in range(self.iteration):
            gamma = self.calculate_Gamma(data, mu, sigma, PI)   # gamma(n,k)
            N = np.sum(gamma,axis=0) # N_k = sum gamma(n)
            PI = N / n
            mu = np.array([1/N[i]*np.sum(data*gamma[:,i].reshape((n,1)),axis=0)  for i in range(self.k)])
            for p in range(self.k):
                # calculate sigma_k
                sigma[p] = 0
                for q in range(n):
                    sigma[p] += np.dot( (data[q].reshape(1,D)-mu[p]).T, (data[q]-mu[p]).reshape(1,D) ) * gamma[q,p]
                sigma[p] = sigma[p] / N[p]
        self.mu = mu
        self.sigma = sigma
        self.PI = PI
    
    def predict(self, data):
        res = self.calculate_Gamma(data, self.mu, self.sigma, self.PI)
        res = np.argmax(res,axis=1)
        return res

data = gaussian_distribution(showplt=True,sample=200)
model = GMM(3)
model.train(data)
label = model.predict(data)

plt.scatter(data[:,0],data[:,1],c=label,marker='.')
plt.scatter(model.mu[:,0],model.mu[:,1],marker='x',color='red')
plt.show()