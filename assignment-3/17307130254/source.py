import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import kmeans2
import time

def Genarate_gussian_distribution(mean,cov,num=500,):
    data= np.random.multivariate_normal(mean,cov,(num))
    return data
def GenarateData(k,means,covs,nums):
    data=[]

    for i in range(k):
        single_data=Genarate_gussian_distribution(means[i],covs[i],nums[i])
        plt.scatter(single_data.transpose()[0],single_data.transpose()[1])
        data.append(single_data)
    for j in means:
        plt.scatter(j[0],j[1],marker='x')
    plt.show()
    plt.clf()
    return data

class EM():
    def __init__(self,k,iterations=50,dimension=2):
        self.k=k
        self.iterations=iterations
        self.dimension=dimension
    
    def init_theta_kmeans(self,k,data,d):        
        centroid, label = kmeans2(data, k)
        counts = np.bincount(label)
        lamda=counts/counts
        mu=centroid
        sigma=[np.eye(d) for _ in range(k)]
        w0 = data[label == 0]
        w1 = data[label == 1]
        w2 = data[label == 2]
        w3 = data[label == 3]
        w4 = data[label == 4]
        
        plt.scatter(w0[:, 0], w0[:, 1], label='cluster 0')
        plt.scatter(w1[:, 0], w1[:, 1], label='cluster 1')
        plt.scatter(w2[:, 0], w2[:, 1],label='cluster 2')
        plt.scatter(w3[:, 0], w3[:, 1],label='cluster 3')
        plt.scatter(w4[:, 0], w4[:, 1],label='cluster 4')
     
        plt.plot(centroid[:, 0], centroid[:, 1], 'k*', label='centroids')
        plt.legend()
        plt.show()
        plt.clf()
        return lamda,mu,sigma
    
    def init_theta(self,k,data,d=2):
        lamda=[1/k for _ in range(k)]
        mu=[data[random.randint(0,len(data)-1)] for _ in range(k)]
        sigma=[np.eye(d) for _ in range(k)]
        return lamda,mu,sigma
    
    def E(self,data,theta,k):
        lamda,mu,sig=theta
        Z = np.zeros((len(data), k))
        p = np.zeros(k)
        for n,x in enumerate(data):
            for i in range(k):                
                det = np.sqrt(np.linalg.det(sig[i]))
                inv = np.linalg.inv(sig[i])
                p[i] = np.exp(-0.5*np.dot(np.dot((x-mu[i]),inv),(x-mu[i]).T))/(2*np.pi*det)
            for i in range(k):
                Z[n,i]=p[i]/p.sum()
        return Z

    def M(self,data,Z):
        d=len(data[0])
        lamda=np.sum(Z,axis=0)
        mu=[1/lamda[i]*np.sum([data[j]*Z[j][i] for j in range(len(data))],axis=0) for i in range(self.k)]
        sigma=[1/lamda[i]*np.sum([ np.dot((data[j].reshape(1,d)-mu[i]).T, (data[j]-mu[i]).reshape(1,d)) * Z[j][i] for j in range(len(data))],axis=0) for i in range(self.k)]
        lamda=lamda/lamda.sum()
        return lamda,mu,sigma

    def ELBO(self,theta,K,Z):
        lamda,mu,sig=theta
        elbo = 0
        for k in range(K):
            det = np.sqrt(np.linalg.det(sig[k]))
            for n,x in enumerate(data): 
                elbo += Z[n][k]* (np.log(lamda[k])-np.log(det)-1/(2*det*det)*np.dot((x-mu[k]).reshape(1,2),(x-mu[k]).reshape(2,1)))
        return elbo

    
    def train(self,data,kmeans=None):
        if kmeans==True:
            theta=self.init_theta_kmeans(self.k,data,len(data[0]))
        else:
            theta=self.init_theta(self.k,data,len(data[0]))
        pre_elbo=0
        #print(theta)
        for iteration in range(self.iterations):
            Z=self.E(data,theta,self.k)
            theta=self.M(data,Z)
            #if iteration<=10:
            #    predict(theta,data,self.k,iteration)
            
            elbo=self.ELBO(theta,self.k,Z)
            print(elbo)
            if abs(elbo-pre_elbo)<1e-4:
                print(iteration)
                break
            pre_elbo=elbo
        #    print(theta)
        return theta
    
def predict(theta,data,k,iteration=None):
    lamda,mu,sig=theta
    Z = np.zeros((len(data), k))
    p = np.zeros(k)
    for n,x in enumerate(data):
        for i in range(k):                
            det = np.sqrt(np.linalg.det(sig[i]))
            inv = np.linalg.inv(sig[i])
            p[i] = np.exp(-0.5*np.dot(np.dot((x-mu[i]),inv),(x-mu[i]).T))/(2*np.pi*det)
        for i in range(k):
            Z[n,i]=p[i]/p.sum()
    Z = np.argmax(Z,axis=1)
    color={0:'b',1:'c',2:'y',3:'g',4:'r'}
    for i,x in enumerate(data):
        plt.scatter(x[0],x[1],color=color[Z[i]])
    for i in mu:
        plt.scatter(i[0],i[1],marker='x')
    if iteration!=None:
        wpath='dataD_kmeans_init_'+str(iteration)+'.png'
        plt.savefig(wpath)
        plt.clf()
    else:
        wpath='dataD_random_init_finish_2.png'
        plt.savefig(wpath)
        plt.clf()
    return Z


mean = [[1,0],[4,2],[2,5]]
cov = [[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]]
#data=GenarateData(3,mean,cov,[500,500,500])
#data=np.array(data).reshape(-1,2)

k=3
model=EM(k,50,3)
theta=model.train(data,True)
label=predict(theta,data,3)
