import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

mean = np.array([[2, 5], [5, 2], [6, 6]])
cov = np.array([
    [[1, -0.5],
     [-0.5, 1]],
    [[0.2, 0],
     [0, 2]],
    [[1, 0.5],
     [0.5, 1]]
])


#第一部分生成数据
class DATA:
    def __init__(self, mean, cov, n):
        self.mean = mean
        self.cov = cov
        self.n = n

    def GenData(self):
        dist1 = np.random.multivariate_normal(self.mean[0], self.cov[0], self.n[0])
        dist2 = np.random.multivariate_normal(self.mean[1], self.cov[1], self.n[1])
        dist3 = np.random.multivariate_normal(self.mean[2], self.cov[2], self.n[2])
        data = []
        plt.plot(dist1.transpose()[0], dist1.transpose()[1], 'bo')
        plt.plot(dist2.transpose()[0], dist2.transpose()[1], 'bo')
        plt.plot(dist3.transpose()[0], dist3.transpose()[1], 'bo')
        plt.show()
        num = len(dist1) + len(dist2) + len(dist3)
        for i in dist1:
            data.append(str(i[0]) + ' ' + str(i[1]) + '\n')
        for i in dist2:
            data.append(str(i[0]) + ' ' + str(i[1]) + '\n')
        for i in dist3:
            data.append(str(i[0]) + ' ' + str(i[1]) + '\n')
        np.random.shuffle(data)
        fw = open('data.data', 'w')
        fw.write(str(num) + '\n')
        for i in data:
            fw.write(i)
        fw.close()
        return dist1, dist2, dist3

    def InputData(self):
        fr = open('data.data', 'r')
        lines_num = int(fr.readline())
        x = np.empty((lines_num, 2))
        t = np.empty((lines_num, 1))
        for i in range(lines_num):
            (x[i][0], x[i][1]) = fr.readline().split()
        fr.close()
        return lines_num, x


#第二部分高斯模型
class GMM:
    def __init__(self, dim, k, iteration):
        self.dim = dim
        self.k = k
        self.iteration = iteration
        self.pi = np.empty((k, 1))
        self.mu = np.empty((k, dim))
        self.sigma = np.empty((self.k, dim, dim))
        self.gamma = None

    #第一种初始化方法
    def initmethod1(self, n, x):
        self.pi = np.ones(self.k, dtype=np.float) / self.k         #各类的先验概率
        self.mu = np.array([x[np.random.randint(0, len(x))] for i in range(self.k)], dtype=np.float)
        self.sigma = np.full((self.k, self.dim, self.dim), np.identity(self.dim), dtype=np.float)

    #第二种初始化方法:kmeans++
    def initmethod2(self, n, x):
        #随机选取第一个聚类中心
        self.mu[0] = x[np.random.randint(0, len(x))]
        p = np.empty(n)
        for q in range(1, self.k, 1):
            for i in range(n):
                d = np.array([np.linalg.norm(x[i] - self.mu[j]) for j in range(q)])
                p[i] = d[np.argmin(d)]
            p_sum = np.sum(p)
            #轮盘法依次选择k-1个聚类中心
            select = np.random.rand()
            for i in range(n):
                select = select - p[i]/p_sum
                if select < 0:
                    self.mu[q] = x[i]

        pi = np.zeros((self.k, 1))
        mu = np.zeros((self.k, self.dim))
        sigma = np.zeros((self.k, self.dim, self.dim))
        t = np.empty(n, dtype=int)
        while(True):
            for i in range(n):
                d = np.array([np.linalg.norm(x[i]-self.mu[p]) for p in range(self.k)])
                t[i] = np.argmin(d)        #找到最近的聚类中心
                pi[t[i]] += 1
                mu[t[i]] += x[i]
            for i in range(self.k):
                if pi[i] == 0:
                    mu[i] = self.mu[i]
                else:
                    mu[i] = mu[i]/pi[i]
            if((mu==self.mu).all()):       #收敛条件
                break
            else:
                self.mu = mu
                self.pi = pi/n
            pi = np.zeros((self.k, 1))
            mu = np.zeros((self.k, self.dim))
        #根据初始分类计算sigma
        for i in range(n):
            sigma[t[i]] += np.dot((x[i] - self.mu[t[i]])[:, None], (x[i] - self.mu[t[i]])[None, :])
        for i in range(self.k):
            self.sigma[i] = sigma[i]/pi[i]
            if np.linalg.det(self.sigma[i]) == 0:
                self.sigma[i] = np.identity(self.dim)

    #计算gamma
    def Gamma(self, n, x):
        p_x_n = np.zeros(self.k)
        gamma = np.empty((n, self.k))
        for j in range(n):
            for m in range(self.k):
                p_x_n[m] = self.pi[m] * st.multivariate_normal.pdf(x[j], mean=self.mu[m], cov=self.sigma[m])
            for m in range(self.k):
                gamma[j][m] = p_x_n[m] / np.sum(p_x_n)
        return gamma

    #训练获得参数
    def train(self, n, x, method):     #method标识初始化方法
        if method == 1:
            self.initmethod1(n, x)
        else:
            self.initmethod2(n, x)
        for i in range(self.iteration):     #迭代iteration次
            self.gamma = self.Gamma(n, x)
            N = np.sum(self.gamma, axis=0)
            #计算pi和mu
            self.pi = N / n
            self.mu = np.array([1 / N[p] * np.sum(x * self.gamma[:, p].reshape((n, 1)), axis=0) for p in range(self.k)])
            #计算sigma
            for m in range(self.k):
                sum_sigma = np.zeros((self.k, self.dim, self.dim))
                for j in range(n):
                    sum_sigma[m] += self.gamma[j][m]*np.dot((x[j] - self.mu[m])[:, None], (x[j] - self.mu[m])[None, :])
                self.sigma[m] = sum_sigma[m]/N[m]
        return self.gamma

    #对数据集分类返回标记数组
    def classify(self, n, x):
        t = np.empty(n)
        for j in range(n):
            t[j] = np.argmax(self.gamma[j])
        return t


if __name__ == '__main__':
    n = np.array([200, 300, 400])
    data = DATA(mean, cov, n)
    dist1, dist2, dist3 = data.GenData()
    n, x = data.InputData()
    gmm = GMM(2, 3, 100)
    gmm.train(n, x, 1)
    t = gmm.classify(n, x)
    out1 = dist1.transpose()
    out2 = dist2.transpose()
    out3 = dist3.transpose()
    plt.subplot(121)
    plt.plot(out1[0], out1[1], 'yo')
    plt.plot(out2[0], out2[1], 'bo')
    plt.plot(out3[0], out3[1], 'ro')
    plt.subplot(122)
    for i in range(n):
        if t[i] == 0:
            plt.plot(x[i][0], x[i][1], 'go')
        elif t[i] == 1:
            plt.plot(x[i][0], x[i][1], 'co')
        elif t[i] == 2:
            plt.plot(x[i][0], x[i][1], 'mo')
        else:
            plt.plot(x[i][0], x[i][1], 'ko')
    plt.show()
