from matplotlib import pyplot as plt
import numpy as np
from scipy import stats as st


##################################################### Create Data ######################################################
mean1 = np.array([1,3])
mean2 = np.array([0,0])
mean3 = np.array([4,1])
cov = np.identity(2)


f1xy = np.random.multivariate_normal(mean1,cov,1000)
f2xy = np.random.multivariate_normal(mean2,cov,1000)
f3xy = np.random.multivariate_normal(mean3,cov,1000)

mydata = np.vstack((f1xy,f2xy,f3xy))
np.random.shuffle(mydata)
plt.scatter(mydata.T[0],mydata.T[1],marker='2')

plt.title("Unprocessed Data")
plt.show()

##################################################### GMM Model ######################################################

class myGMMmodel():
    def __init__(self , k):
        self.k = k
        self.D = 2
        self.pi = 0
        self.sigma = 0
        self.miu = 0

    def E_Step(self,x,pi,sigma,miu):
        k = self.k
        D = self.D
        n = x.shape[0]
        gamma = np.zeros((n, k))
        N = np.zeros(k)
        for p in range(n):
            for q in range(k):
                N[q] = pi[q] * st.multivariate_normal.pdf(x[p], miu[q], sigma[q])
            sumN = np.sum(N)
            for q in range(k):
                gamma[p, q] = N[q] / sumN
        return gamma

    def M_Step(self,gamma,x,pi,sigma,miu):
        k = self.k
        D = self.D
        n = x.shape[0]
        Nk = np.sum(gamma,0)

        pi = Nk / n
        for i in range(k):
            miu[i] = np.zeros((2,))
            for j in range(n):
                miu[i] += (gamma[j , i] * x[j])
            miu[i] = miu[i] / Nk[i]

        for j in range(k):
            sigma[j] = np.zeros((D,D))
            for i in range(n):
                sigma[j] += gamma[i, j] * np.dot((x[i]-miu[j]).reshape(D,1), ((x[i]-miu[j]).reshape(D,1)).T)
            sigma[j] = sigma[j] / Nk[j]

        return pi,sigma,miu

    def train(self,x):
        k = self.k
        D = self.D
        n = x.shape[0]
        epoch = 50

        pi = np.ones(k,dtype = np.float) / k
        miu = np.array(np.array([x[np.random.randint(0,n)] for i in range(k)]),dtype=np.float)
        sigma = np.zeros((k,D,D))
        for i in range(k):
            sigma[i] = np.identity(2)

        for e in range(epoch):
            gamma = self.E_Step(x,pi,sigma,miu)
            pi,sigma,miu = self.M_Step(gamma,x,pi,sigma,miu)
            print('epoch = ', e)

        self.miu = miu
        self.sigma = sigma
        self.pi = pi



model = myGMMmodel(3)
model.train(mydata)


plot2 = np.argmax(model.E_Step(mydata,model.pi,model.sigma,model.miu),1)
plt.scatter(mydata[:, 0], mydata[:, 1], c = plot2, marker='2')
plt.scatter(model.miu[:, 0], model.miu[:, 1], marker='D', color='blue')
plt.scatter([0,1,4],[0,3,1],marker='D',color='red')
plt.title("processed Data")
plt.show()