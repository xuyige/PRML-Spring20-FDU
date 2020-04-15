import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import time

def plot_scatter(data1,data2,data3):
    
    dist1=data1.transpose()
    dist2=data2.transpose()
    dist3=data3.transpose()
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(dist1[0],dist1[1],dist1[2])
    ax1.scatter3D(dist2[0],dist2[1],dist2[2])
    ax1.scatter3D(dist3[0],dist3[1],dist3[2])

def GenarateData(n,num1,num2,num3):

    mean1=3*np.eye(n)[0]
    mean2=3*np.eye(n)[1]
    mean3=3*np.eye(n)[2]
    cov1=np.eye(n)
    cov2=np.eye(n)
    cov3=np.eye(n)
    
    data=[]
    data1= np.random.multivariate_normal(mean1,cov1,(num1))
    data2= np.random.multivariate_normal(mean2,cov2,(num2))
    data3= np.random.multivariate_normal(mean3,cov3,(num3))
    plot_scatter(data1,data2,data3)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    dataset=[]
    label=0
    for d in data:
        label+=1
        for node in d:
            dataset.append(np.append(node,label))
    #dataset=np.array(dataset)
    #dataset=dataset.shuffle()
    random.shuffle(dataset)
    all=num1+num2+num3
    return dataset[0:int(all*0.7)],dataset[int(all*0.7):],data

class LinearGenerativeModel:
    def __init__(self,n,m):
        self.n=n
        self.nodes=m
        self.pi = np.zeros ( (3) )
        self.mu = np.zeros ( (3,n) )
        self.sigma = np.zeros ( (3,n,n) )
    def train(self,data):
        start=time.time()
        num=np.zeros(3)
        for node in data:
            label=int(node[self.n])-1
            num[label]+=1
        self.pi=num/sum(num)
        #print(self.pi)
        
        for node in data:
            label=int(node[self.n])-1
            for i in range(self.n):
                self.mu[label][i]+=node[i]
        for i in range(3):
            self.mu[i]/=num[i]
        #print(self.mu)
        
        for node in data:
            label=int(node[self.n])-1
            x=node[:self.n]
            y=x-self.mu[label]
            self.sigma[label]+=np.dot(y.reshape(self.n,1),y.reshape(1,self.n))
        for i in range(3):
            self.sigma[i]/=num[i]
        end=time.time()
        return end-start
        #print(self.sigma)
        
    def test(self,test):    
        prob=np.zeros(3)
        sum=0
        for node in test:
            for i in range (3):
                prob[i] = stats.multivariate_normal(self.mu[i],self.sigma[i]).pdf(node[:self.n])*self.pi[i]
            if np.argmax(prob)!=int(node[self.n])-1:
                sum+=1 
        return 1-sum/self.nodes



def softmax(x):
    x = np.asmatrix(x)
    x=x.reshape(1,3)
    x = x - np.max(x, axis=1)
    return np.exp(x) / np.sum(np.exp(x), axis=1)

class LinearDiscriminativeModel:
    def __init__(self,n,m):
        self.n=n
        self.nodes=m
        self.w=np.asmatrix(np.zeros((n+1,3)))
        self.lr=0.01
    def train(self,train):
        start=time.time()
        for node in train:
            label=int(node[self.n])-1
            xn=np.append(node[:self.n],1).reshape(self.n+1,1)
            yn=softmax(self.w.T*xn).reshape(1,3)
            y=np.argmax(yn)     
            tn=np.eye(3)[label]
            self.w+=self.lr*np.dot(xn,tn-yn)
        end=time.time()
        return end-start
            
    def test(self,test):
        sum=0
        for node in test:
            label=int(node[self.n])-1
            xn=np.append(node[:self.n],1).reshape(self.n+1,1)
            yn=softmax(self.w.T*xn).reshape(1,3)
            y=np.argmax(yn)
            if y!=label:
                sum+=1
        return 1-sum/self.nodes

    def getw(self):
        return self.w
        

            
def plot():
    fig = plt.figure()
    ax = plt.axes(projection='3d')


    dist1=data[0].transpose()
    dist2=data[1].transpose()
    dist3=data[2].transpose()

    a=np.array(w[0])[0]
    b=np.array(w[1])[0]
    c=np.array(w[2])[0]
    X = np.linspace(-3,3,30)
    Y = np.linspace(-3,3,30)
    X, Y = np.meshgrid(X, Y)
    for i in range(3):
        Z1=-X*a[0]/a[2]-Y*a[1]/a[2]-a[3]/a[2]

    ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    for i in range(3):
        Z2=-X*b[0]/b[2]-Y*b[1]/b[2]-b[3]/b[2]
    #ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    for i in range(3):
        Z3=-X*c[0]/c[2]-Y*c[1]/c[2]-c[3]/c[2]
    #ax.plot_surface(X, Y, Z3, rstride=1, cstride=1, cmap='viridis', edgecolor='none')



    ax.scatter3D(dist1[0],dist1[1],dist1[2])
    ax.scatter3D(dist2[0],dist2[1],dist2[2])
    ax.scatter3D(dist3[0],dist3[1],dist3[2])


n=3
train,test,data=GenarateData(n,100,100,100)
print('train nodes: '+str(len(train))+'     test nodes: '+str(len(test)))
m=len(test)

LG=LinearGenerativeModel(n,m)
print('Generative Model train time: '+ str(LG.train(train)))
print('Generative Model test Accuracy: '+ str(LG.test(test)))

LD=LinearDiscriminativeModel(n,m)
print('Discriminative Model train time: '+ str(LD.train(train)))
w=LD.getw()
print('Discriminative Model test Accurarcy: '+ str((LD.test(test))))
w=w.T

plot()


