import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.stats import multivariate_normal
import sys
np.random.seed(0)


def create_toy_data(samples,vara,varb,varc,add_outliers=False, add_class=False):
    x0 = np.random.normal(scale=vara,size=2*samples).reshape(-1, 2) - 1
    x1 = np.random.normal(scale=varb,size=2*samples).reshape(-1, 2) + 1.
    if add_outliers:
        x_1 = np.random.normal(size=10).reshape(-1, 2) + np.array([5., 10.])
        return np.concatenate([x0, x1, x_1]), np.concatenate([np.zeros(25), np.ones(30)]).astype(np.int)
    if add_class:
        x2 = np.random.normal(scale=varc,size=2*samples).reshape(-1, 2) + 3.
        return np.concatenate([x0, x1, x2]), np.concatenate([np.zeros(samples), np.ones(samples), 2 + np.zeros(samples)]).astype(np.int)
    return np.concatenate([x0, x1]), np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)

class SoftmaxRegression:
    
    def _softmax(self,a):
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=100, learning_rate:float=0.1):
       
        if t.ndim == 1:
            t = encode(t)
        self.n_classes = np.size(t, 1)
        W = np.zeros((np.size(X, 1), self.n_classes))
       
        for _ in range(max_iter):
            W_prev = np.copy(W)
            y = self._softmax(X @ W)
            grad = X.T @ (y - t)
            W -= learning_rate * grad
            if np.allclose(W, W_prev):
                break
        self.W = W

    def proba(self, X:np.ndarray):
      
        return self._softmax(X @ self.W)

    def classify(self, X:np.ndarray):
       
        return np.argmax(self.proba(X), axis=-1)

class GDA:
    def fit(self,X:np.ndarray,t:np.ndarray):
        N0 = sum(t == 0)
        N1 = sum(t == 1)
        N2 = sum(t == 2)
        self.p0 = p0 = N0/(N0+N1+N2)
     
        self.p1 = p1 = N1/(N1+N0+N2)
        self.p2 = p2 = N2/(N1+N0+N2)
        y0 = (t==0)
    
        self.u0 = np.mean(X[y0],axis=0)
        y1 = (t==1)
     
        self.u1 = np.mean(X[y1],axis=0)
        y2 = (t==2)
 
        self.u2 = np.mean(X[y2],axis=0)
 
        self.co = p0*np.cov(X[y0>0,:].T)+p1*np.cov(X[y1>0,:].T)+p2*np.cov(X[y2>0,:].T)
        #print(self.co)
      
    def proba(self,X:np.ndarray):
        self.P0  = np.mat(self.p0*multivariate_normal.pdf(X,self.u0,self.co)); 
 
        self.P1  = np.mat(self.p1*multivariate_normal.pdf(X,self.u1,self.co)); 

        self.P2  = np.mat(self.p2*multivariate_normal.pdf(X,self.u2,self.co));
        self.p =  np.concatenate((self.P0,self.P1,self.P2),axis=0)
        self.p = self.p.T
        return self.p

    def classify(self, X:np.ndarray):
        return np.argmax(self.proba(X), axis=1)   
    
def encode(a:np.ndarray):
    a=np.eye(3)[a] 
    return a

def transform(a:np.ndarray):
    n = np.size(a,0)
    b = np.ones([1,n]) 
    a = np.insert(a, 0, values=b, axis=1)
    return a




def main(argv):
    x_train, y_train = create_toy_data(int(argv[1]),int(argv[2]),int(argv[3]),int(argv[4]),add_class=True)
    x1, x2 = np.meshgrid(np.linspace(-5, 10, 100), np.linspace(-5, 10, 100))
    x = np.array([x1, x2]).reshape(2, -1).T
    X_train = transform(x_train)
    X = transform(x)
    model = GDA()
    model2 = SoftmaxRegression()
    model2.fit(X_train, y_train, max_iter=10000, learning_rate=0.01)
    model.fit(x_train, y_train)
    y = model.classify(x)
    y2=model2.classify(X)


    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.contourf(x1, x2, y.reshape(100, 100), alpha=0.2, levels=np.array([0., 0.5, 1.5, 2.]))
    plt.xlim(-5, 10)
    plt.ylim(-5, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('GDA')
    plt.savefig('./testgg.jpg')
    plt.show()

    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.contourf(x1, x2, y2.reshape(100, 100), alpha=0.2, levels=np.array([0., 0.5, 1.5, 2.]))
    plt.xlim(-5, 10)
    plt.ylim(-5, 10)
    plt.title('Softax Regression')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('./testd.jpg')
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
