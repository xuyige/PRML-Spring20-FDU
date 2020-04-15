"""
#创建数据集
mean = [[-1, 0], [1, 1], [0, 2]]
cov = [[[0.1,0],[0,0.1]],[[0.1,0],[0,0.1]],[[0.1,0],[0,0.1]]]
createData(mean,cov)

#划分数据集
x_train,y_train,x_test,y_test = loadData(0.8) #按8:2划分训练集与测试集

#训练并测试生成模型
generative_model = GModel()
generative_model.train(x_train,y_train)
acc = generative_model.test(x_test,y_test)

#训练并测试判别模型
disc_model = DModel()
disc_model.train(x_train,y_train,x_test,y_test,mini_batch = 64,epoch = 15,alpha = 0.3)
"""
import numpy as np
import matplotlib.pyplot as plt
import collections
from scipy.stats import multivariate_normal 
import math

def createData(means,covs):
    f = open("data.data","w")
    label_num = len(means) #共lable_num个类
    data_scale = 500 #每类数据点数
    data_set = []
    total_data = 0
    for i in range(label_num):
        total_data += data_scale
        cordinate_set_i = np.random.multivariate_normal(means[i],covs[i],data_scale).T
        label_set_i = np.full((1,data_scale),i)
        data_set_i = np.concatenate((cordinate_set_i,label_set_i))
        data_set = data_set_i if i == 0 else np.concatenate((data_set,data_set_i),axis = 1)      
        plt.plot(cordinate_set_i[0],cordinate_set_i[1],'x'); plt.axis('equal');
    
    #打乱数据
    np.random.shuffle(data_set.T)
    
    #保存数据到文件
    for j in range(total_data):
        data = str(round(data_set[0][j],3))+" "+str(round(data_set[1][j],3))+" "+str(int(data_set[2][j]))+"\n" #保留三位小数
        f.write(data)
    f.close()
    
    #可视化
    plt.show()
    
def loadData(ratio = 0.9):
    #数据已经被打乱，需要划分训练集和测试集
    f = open("data.data","r")
    x,y = [],[]
    for line in f.readlines():
        items = line.strip("\n").split(" ")
        x.append(np.array([float(items[0]),float(items[1])]));y.append(int(items[2]))
    limit = round(ratio*len(x))
    x_train,y_train,x_test,y_test = np.array(x[:limit]),np.array(y[:limit]).reshape(-1,1),np.array(x[limit:]),np.array(y[limit:]).reshape(-1,1)
    return x_train,y_train,x_test,y_test

def __init__():
    mean = [[0,0],[1,1],[2,2]]
    cov = [[[0.1,0],[0,0.1]],[[0.1,0],[0,0.1]],[[0.1,0],[0,0.1]]]
    createData(mean,cov)
    
    x_train,y_train,x_test,y_test = loadData(0.8) #按9:1划分训练集与测试集
    generative_model = GModel()
    generative_model.train(x_train,y_train)
    acc = generative_model.test(x_test,y_test)
    print(acc)

class GModel: #线性生成模型
    def __init__(self):
        pass
    
    def softmax(self,x):
        x = np.asmatrix(x)
        ex = np.exp(x - np.max(x))
        return ex / ex.sum()
    
    def train(self,x,y):
        self.N = x.shape[0]
        self.dimension = x.shape[1]
        self.class_counter = collections.defaultdict(int)
        self.sigma = np.zeros((self.dimension,self.dimension))
        self.means = {}
        y = y.reshape(y.shape[0],)
        
        self.class_num = max(y) + 1
        for i in range(self.N):
            self.class_counter[y[i]] += 1
            if y[i] not in self.means.keys(): 
                self.means[y[i]] = x[i]
            else:
                self.means[y[i]] += x[i]
        for c in self.class_counter.keys():
            self.means[c] /= self.class_counter[c]
        for i in range(self.N):
            self.sigma += np.dot((x[i] - self.means[y[i]]).reshape(1,2).T,(x[i] - self.means[y[i]]).reshape(1,2))
        
        self.sigma /= self.N
        self.istrained = 1
        
    def predict(self,x):
        if not self.istrained:return 
        possibility = np.zeros((self.class_num,1))
        for i in range(self.class_num):
            possibility[i] = multivariate_normal(self.means[i],self.sigma).pdf(x) * self.class_counter[i] / self.N
        possibility = self.softmax(np.log(possibility))
        return possibility.argmax()
        
    def test(self,x_test,y_test):
        #输出准确率
        y_test = y_test.reshape(y_test.shape[0],)
        self.confuse_matrix = np.zeros((self.class_num,self.class_num))
        acc = 0
        for i in range(len(x_test)):
            self.confuse_matrix[y_test[i]][self.predict(x_test[i])] += 1
            if self.predict(x_test[i]) == y_test[i]:
                acc += 1
        print("generative model accuracy: ",round(acc / len(x_test),3)) 
        print("confusing matrix:\n",self.confuse_matrix,"\n")
        return acc



class DModel: #线性判别模型
    def softmax(self,x):
        x = np.asmatrix(x)
        ex = np.exp(x - np.max(x, axis=1))
        return ex / np.sum(ex, axis=1)

    def gradients(self,x,y):
        return x.T.dot(y - self.softmax(self.w.T.dot(x.T).T))
    
    def train(self,x,y,x_test,y_test,mini_batch = 128,epoch = 100,alpha = 0.1): #小样本随机梯度下降，默认为随机梯度下降      
        self.N = x.shape[0]
        self.dimension = x.shape[1]
        self.class_num = np.max(y) + 1
        self.w = np.zeros((self.dimension,self.class_num))
        
        #构造one-hot y
        y_0 = np.zeros((self.N,self.class_num))
        for i in range(self.N):
            y_0[i][y[i][0]] = 1
        y = y_0
    
        #分成多个批次
        batch_num = math.ceil(self.N / mini_batch)
        x_in_batch = np.array([x[mini_batch*i:mini_batch*(i+1)] for i in range(batch_num)])
        y_in_batch = np.array([y[mini_batch*i:mini_batch*(i+1)] for i in range(batch_num)])
        print("discriminal model training...")
        print("Epoch\t","Acc\t")
        for i in range(1,epoch+1):
            for j in range(x_in_batch.shape[0]):
                w_grads = self.gradients(x_in_batch[j],y_in_batch[j])
                #更新参数
                self.w += (alpha * w_grads) / x_in_batch[j].shape[0]
        
            acc = self.test(x_test,y_test)
            print(i,"\t",acc)
        print("confusing matrix:\n",self.confuse_matrix,"\n")
            
        
    def predict(self,x):
        x = np.asmatrix(x)
        possibility_array = self.softmax(self.w.T.dot(x.T).T)
        return possibility_array.argmax()
        
    def test(self,x_test,y_test):
        #输出准确率
        acc = 0
        self.confuse_matrix = np.zeros((self.class_num,self.class_num))
        for i in range(len(x_test)):
            self.confuse_matrix[y_test[i][0]][self.predict(x_test[i])] += 1
            if self.predict(x_test[i]) == y_test[i]:
                acc += 1
        return round(acc / len(x_test),3)

if __name__ == "__main__":
    mean = [[-1, 0], [1, 1], [0, 2]]
    cov = [[[0.1,0],[0,0.1]],[[0.1,0],[0,0.1]],[[0.1,0],[0,0.1]]]
    createData(mean,cov)
    
    x_train,y_train,x_test,y_test = loadData(0.8)
    generative_model = GModel()
    generative_model.train(x_train,y_train)
    acc = generative_model.test(x_test,y_test)

    disc_model = DModel()
    disc_model.train(x_train,y_train,x_test,y_test,mini_batch = 64,epoch = 15,alpha = 0.9)