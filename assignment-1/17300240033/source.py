import os
os.sys.path.append('..')
# use the above line of code to surpass the top module barrier
from handout import *
import numpy as np
import matplotlib.pyplot as plt

train = []
test = []
ans = []

def show_data(label,title):
    x = [[] for i in range(3)]
    y = [[] for i in range(3)]
    for i in range(len(test)):
        x[int(label[i])].append(test[i][0])
        y[int(label[i])].append(test[i][1])
    plt.scatter(x[0],y[0],c='red')
    plt.scatter(x[1],y[1],c='green')
    plt.scatter(x[2],y[2],c='blue')
    plt.title(title,fontsize = 18)
    plt.show()

def generate_data():
    mean1 = [0,0]
    cov1 = [[10,0],[0,10]]
    mean2 = [10,10]
    cov2 = [[10,0],[0,10]]
    mean3 = [20,0]
    cov3 = [[10,0],[0,10]]
    
    num1 = 50
    num2 = 50
    num3 = 50

    lis = []
    for i in range(num1):
        x = np.random.multivariate_normal(mean1,cov1) 
        x = np.append(x,0)
        lis.append(x)
    for i in range(num2):
        x = np.random.multivariate_normal(mean2,cov2) 
        x = np.append(x,1)
        lis.append(x)
    for i in range(num3):
        x = np.random.multivariate_normal(mean3,cov3) 
        x = np.append(x,2)
        lis.append(x)
    
    lis = np.array(lis)
    np.random.shuffle(lis)

    np.savetxt("gauss.data",lis)
    
def get_data():
    global train,test,ans
    data = np.loadtxt("gauss.data")
    x = int(len(data)*0.8)
    train = data[0:x]
    test = data[x:,0:2]
    ans = data[x:,2]
    title = 'correct classification'
    show_data(ans,title)  
    
def get_acc(label):
    m = len(test)
    tot = 0
    for i in range(m):
        if ans[i]==label[i]: tot = tot+1
    acc = 100.0*tot/m
    return acc

#generative model
def generative_model(): 
    n = len(train)
    m = len(test)
    c = [[] for i in range(3)]
    p = [0 for i in range(3)]
    for i in range(n):
        c[int(train[i][2])].append(train[i][0:2])
    mu = np.zeros((3,1,2))
    sig = np.zeros((3,2,2))
    inv = np.zeros((3,2,2))
    det = np.zeros((3))
    for i in range(3):
        p[i]=len(c[i])/n
        for j in range(len(c[i])):
            mu[i] = mu[i]+c[i][j]
        mu[i] = mu[i]/len(c[i])
        for j in range(len(c[i])):
            sig[i] = sig[i]+np.dot((c[i][j]-mu[i]).T,(c[i][j]-mu[i]))
        sig[i] = sig[i]/len(c[i])
        inv[i] = np.linalg.inv(sig[i])
        det[i] = np.linalg.det(sig[i]) 
        det[i] = np.sqrt(det[i])

    pi = np.pi
    label = np.zeros((m))
    for i in range(m):
        pos = [0 for k in range(3)]
        for j in range(3):
            lk = np.exp(-0.5*np.dot(np.dot((test[i]-mu[j]),inv[j]),(test[i]-mu[j]).T))/(2*pi*det[j])
            pos[j] = p[j]*lk
        if pos[0]>pos[1] and pos[0]>pos[2]:
            label[i]=0
        elif pos[1]>pos[0] and pos[1]>pos[2]:
            label[i]=1
        else :
            label[i]=2

    title = 'generative model'
    show_data(label,title)
    acc = get_acc(label)
    print('generative model accuracy: %.2f%%'%acc)

#discriminative model
def sigmoid(z):
    if z >= 0:
        return 1/(1+np.exp(-z))
    else:
        return np.exp(z)/(1+np.exp(z))

def discriminative_model():
    lr = 0.01
    epoch = 500
    
    n = len(train)
    m = len(test)
    w = np.random.randn(3,2)
    x = np.array(train[:,:2])
    y = np.zeros((n,3))
    b = np.random.randn(3) 
    t = np.zeros((n,3))
    for i in range(n):
        t[i][int(train[i][2])] = 1

    for i in range(epoch):
        for j in range(n):
            y[j] = np.dot(w,x[j])+b
            for k in range(3): 
                y[j][k]=sigmoid(y[j][k])
        for j in range(n):
            w = w+lr*np.dot((t[j]-y[j]).reshape(3,1),x[j].reshape(1,2))
            b = b+lr*(t[j]-y[j])
                      
    x = np.array(test)
    pos = np.zeros((3))
    label = np.zeros((m))
    for i in range(m):
        pos = np.dot(w,x[i])+b
        if pos[0]>pos[1] and pos[0]>pos[2]:
            label[i]=0
        elif pos[1]>pos[0] and pos[1]>pos[2]:
            label[i]=1
        else :
            label[i]=2
        
    title = 'discriminative model'
    show_data(label,title)
    acc = get_acc(label)
    print('discriminative model accuracy: %.2f%%'%acc)

if __name__ == '__main__':
    #generate_data()
    get_data()
    generative_model()
    discriminative_model()
    