import numpy as np
import matplotlib.pyplot as plt
import random

def normpdf ( x , mu , sigma ):
    return 1. / (2*np.pi*(np.linalg.det(sigma)**0.5)) * np.exp ( -0.5 * np.dot ((x-mu).reshape((1,2)),np.dot(np.linalg.inv(sigma),(x-mu).reshape(2,1))) )

class model:
    def dis ( self , j , l ):
        return (self.point[j][0]-self.mu[l][0])*(self.point[j][0]-self.mu[l][0])+(self.point[j][1]-self.mu[l][1])*(self.point[j][1]-self.mu[l][1]) 

    def kmeans (self):
        self.mu = np.zeros ( (self.k,2) )
        self.mu[0] = self.point[np.random.randint(0,self.n)]
        d = np.zeros ( (self.n) )
        for i in range (self.k):
            if i==0: continue
            for j in range (self.n):
                d[j] = min([self.dis(j,l) for l in range (i)])
            d = d / sum ( d )
            sed = random.random()
            now = 0.0
            for j in range (self.n):
                now += d[j]
                if now >= sed or j == self.n - 1:
                    self.mu[i] = self.point[j]
                    break
        #print ( self.mu )
        for epoch in range (2):
            nmu = np.zeros((self.k,2))
            cnt = np.zeros((self.k))
            for i in range(self.n):
                pos = np.argmin ([self.dis(i,l) for l in range(self.k)])
                nmu[pos] += self.point[i]
                cnt[pos] += 1
            for i in range(self.k):
                self.mu[i] = nmu[i] / cnt[i]

    def __init__ (self , dim , num , data ):
        self.k = dim
        self.pi = np.full ( (self.k) , 1.0/dim )
        self.n = num
        self.point = data
        self.kmeans () #calulate mu
        #self.mu = [data[np.random.randint(0,num)] for i in range(dim)]
        self.sigma = np.zeros ( (self.k,2,2) )
        for i in range (self.k):
            self.sigma[i] = np.diag ( np.full(2,0.1) )

    def E_step ( self ):
        self.gamma = np.zeros ( (self.n,self.k) )
        for i in range ( self.n ):
            for j in range ( self.k ):
                self.gamma[i][j] = self.pi[j] * normpdf ( self.point[i] , self.mu[j] , self.sigma[j] )
            self.gamma[i] = self.gamma[i] / np.sum(self.gamma[i])
    
    def M_step ( self ):
        N = self.gamma.sum ( axis = 0 )
        self.pi = N / np.sum ( N )
        self.mu = np.zeros ( (self.k,2) )
        self.sigma = np.zeros ( (self.k,2,2) )
        for i in range ( self.n ):
            for j in range ( self.k ):
                self.mu[j] += self.gamma[i][j] * self.point[i]
        for j in range ( self.k ):
            self.mu[j] /= N[j]
        for i in range ( self.n ):
            for j in range ( self.k ):
                self.sigma[j] += self.gamma[i][j] * np.dot ( (self.point[i]-self.mu[j]).reshape((2,1)) , (self.point[i]-self.mu[j]).reshape((1,2)) )
        #print ( self.sigma )
        for j in range ( self.k ):
            self.sigma[j] /= N[j]


    def train ( self ):
        self.E_step ()
        self.M_step ()

    def output ( self , draw ):
        self.E_step ()

        pred = np.argmax ( self.gamma , axis=1 )
        #print ( self.gamma[0] , pred[0] )

        if draw == 1:
            plt.cla ()
            out = [[[],[]] for i in range ( self.k )]
            for i in range ( self.n ):
                out[pred[i]][0].append ( self.point[i][0] )
                out[pred[i]][1].append ( self.point[i][1] )
            
            for i in range ( self.k ):
                plt.scatter ( out[i][0] , out[i][1] , marker='o' )
            for i in range ( self.k ):
                plt.scatter ( self.mu[i][0] , self.mu[i][1] , marker = 'x' )
            plt.show ()

        return pred
