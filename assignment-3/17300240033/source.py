import numpy as np
import matplotlib.pyplot as plt

data = []
label = []

def show_data(K):
    x = [[] for i in range(K)]
    y = [[] for i in range(K)]
    clr = ['red','green','blue','yellow','black']
    for i in range(len(data)):
        x[int(label[i])].append(data[i][0])
        y[int(label[i])].append(data[i][1])
    for i in range(K):
        plt.scatter(x[i],y[i],c=clr[i])
    plt.title('result(K=%d)'%(K),fontsize = 18)
    plt.show()

def generate_data():
    mean1 = [0,0]
    cov1 = [[10,0],[0,10]]
    mean2 = [10,10]
    cov2 = [[10,0],[0,10]]
    mean3 = [20,0]
    cov3 = [[10,0],[0,10]]
    
    num1 = 100
    num2 = 100
    num3 = 100

    lis = []
    for i in range(num1):
        x = np.random.multivariate_normal(mean1,cov1) 
        lis.append(x)
    for i in range(num2):
        x = np.random.multivariate_normal(mean2,cov2) 
        lis.append(x)
    for i in range(num3):
        x = np.random.multivariate_normal(mean3,cov3) 
        lis.append(x)
    
    lis = np.array(lis)
    np.random.shuffle(lis)
    np.savetxt("gauss.data",lis)   
    
def get_data():
    global data
    data = np.loadtxt("gauss.data")
    x = []
    y = []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    plt.scatter(x,y,c='purple')
    plt.title('dataset',fontsize = 18)
    plt.show()
    
def lk(x,mu,sig):
    det = np.sqrt(np.linalg.det(sig))
    inv = np.linalg.inv(sig)
    p = np.exp(-0.5*np.dot(np.dot((x-mu),inv),(x-mu).T))/(2*np.pi*det)
    return p
    
def cluster(K):
    N = len(data)
    x = data
    pi = np.random.rand(K)+0.1
    mu = np.random.rand(K,2)*20
    sig = np.zeros((K,2,2))
    r = np.zeros((N,K))
    Nk = np.zeros((K))
    for i in range(K):
        sig[i][0][0] = np.random.rand()*20+1
        sig[i][1][1] = sig[i][0][0]
    
    print("start")
    epoch = 200
    Q_pre = 0
    for step in range(epoch):
        # E step
        for n in range(N):
            tot = 0
            for k in range(K):
                r[n][k] = pi[k]*lk(x[n],mu[k],sig[k])
                tot += r[n][k]
            for k in range(K):
                r[n][k] /= tot
        # M step
        for k in range(K):
            Nk[k] = 0
            for n in range(N):
                Nk[k] += r[n][k]
        for k in range(K):
            pi[k] = Nk[k]/N
            mu[k] = 0
            for n in range(N):
                mu[k] += r[n][k]*x[n]
            mu[k] /= Nk[k]
            sig[k] = 0
            for n in range(N):
                sig[k] += r[n][k]*np.dot((x[n]-mu[k]).reshape(2,1),(x[n]-mu[k]).reshape(1,2))
            sig[k] /= Nk[k]
        # evaluate
        Q = 0
        for k in range(K):
            Q += Nk[k]*np.log(pi[k])
            det = np.sqrt(np.linalg.det(sig[k]))
            for n in range(N): 
                Q += r[n][k]*(np.log(1/np.sqrt(2*np.pi))-np.log(det) \
                    -1/(2*det*det)*np.dot((x[n]-mu[k]).reshape(1,2),(x[n]-mu[k]).reshape(2,1)))
        if step%10==0: 
            print(step," steps: Q=",float(Q),sep='')
        if step>0 and np.abs(Q-Q_pre)<1e-4: 
            break
        Q_pre = Q
    
    print("finish")
    for n in range(N):
        mx = 0
        cls = 0
        for k in range(K):
            if r[n][k]>mx:
                mx = r[n][k]
                cls = k
        label.append(cls)
    
    show_data(K)
    

if __name__ == '__main__':
    #generate_data()
    get_data()
    cluster(3)
    
    