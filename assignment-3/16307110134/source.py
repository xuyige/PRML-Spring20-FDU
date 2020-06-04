import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

def createData(means, covs, data_scales = None):
    f = open("data.data", "w")
    label_num = len(means)  # 共lable_num个类

    if data_scales is None:
        #默认一类500个数据点
        data_scales = [500 for i in range(label_num)]
    else:
        #未指定类的数据点个数一律用500补齐
        data_scales = data_scales[:min(len(data_scales), label_num)] + \
                      [500 for i in range(min(len(data_scales), label_num), label_num)]

    data_set = []
    total_data = 0
    for i in range(label_num):
        total_data += data_scales[i]
        cordinate_set_i = np.random.multivariate_normal(means[i], covs[i], data_scales[i]).T
        label_set_i = np.full((1, data_scales[i]), i)
        data_set_i = np.concatenate((cordinate_set_i, label_set_i))
        data_set = data_set_i if i == 0 else np.concatenate((data_set, data_set_i), axis=1)

    # 保存数据到文件
    for j in range(total_data):
        data = str(round(data_set[0][j], 3)) + " " + str(round(data_set[1][j], 3)) + " " + str(
            int(data_set[2][j])) + "\n"  # 保留三位小数
        f.write(data)
    f.close()

def loadData():
    f = open("data.data", "r")
    x, y = [], []
    for line in f.readlines():
        items = line.strip("\n").split(" ")
        x.append(np.array([float(items[0]), float(items[1])]))
        y.append(int(items[2]))
    return np.array(x), np.array(y).reshape(-1, 1)

def printEllipse(ax, mean, cov, color = "g", shreshold = "95"):
    '''
    作置信椭圆
    '''
    # get eigenvector
    lams, lamvec = np.linalg.eig(cov)
    if shreshold == "99":
        shreshold = 9.210 # 99% confident
    elif shreshold == "90":
        shreshold = 4.605 # 90% confident
    else:
        shreshold = 5.99 # 95% confident

    width = 2 * np.sqrt(shreshold * np.max(lams))
    height = 2 * np.sqrt(shreshold * np.min(lams))
    degree = np.arctan(lamvec[np.argmax(lams)][1] / lamvec[np.argmax(lams)][0])

    e = Ellipse(xy=(mean), width=width, height=height, angle=degree,
                        fill=False, linestyle="-", linewidth=1, color=color)
    ax.add_patch(e)

class GMM:
    def __init__(self, x, mu=None, cov=None, alpha=None, isRandom=False):
        self.k = 3
        self.dim = 2

        if mu:
            self.mu = mu
        else:
            self.mu = np.zeros([self.k, self.dim])
            if isRandom:
                # 随机化选择中心
                self.mu = np.array([x[i] for i in [np.random.choice(range(x.shape[0]), self.k)]])[0]
            else:
                # K-means++ 初始化
                self.mu[0] = x[np.random.choice(range(x.shape[0]))]
                for i in range(1, self.k):
                    dists = np.zeros([x.shape[0], i])
                    for j in range(i):
                        dists[:, j] = np.linalg.norm(x-self.mu[j], axis=1)

                    mins = np.min(dists, axis=1)
                    poss = mins / np.sum(mins)

                    self.mu[i] = x[np.random.choice(range(x.shape[0]), p=poss)]

            # 显示初始选择的数据点
            fig = plt.figure()
            colors = "ygb"
            for i in range(x.shape[0]):
                plt.plot(x[i][0], x[i][1], '.', color=colors[y[i][0]], markersize=2)
            for i in range(self.k):
                plt.plot(self.mu[i][0], self.mu[i][1], "x", color="red", markersize=15)
            plt.title("Initial Centers")

        self.cov = np.array([[[1., 0.], [0., 1.]]] * self.k) if cov is None else cov
        self.alpha = np.ones(self.k) / self.k  if alpha is None else alpha # 默认各高斯分布权值均等

    def sample(self, sample_num):
        samples = np.zeros([sample_num, self.dim])
        for i in range(sample_num):
            j = np.random.choice(range(self.k), p=self.alpha)
            samples[i, :] = np.random.multivariate_normal(self.mu[j], self.cov[j], 1)
        return samples

    def train(self, x, epoch = 10):
        origin_model = GMM(x, mean, cov, data_scales / np.sum(data_scales))

        # save train data to train.txt
        with open("train.txt", "w"): pass

        # EM algorithm
        N = x.shape[0]
        for ep in range(epoch):
            print("Epoch {0} starts".format(ep))
            gamma = np.zeros([N, self.k])
            for i in range(N):
                gamma_sum = np.sum([self.alpha[j] * multivariate_normal.pdf(x[i], self.mu[j], self.cov[j]) for j in range(self.k)])
                for j in range(self.k):
                    gamma[i][j] = self.alpha[j] * multivariate_normal.pdf(x[i], self.mu[j], self.cov[j]) / gamma_sum
            for j in range(self.k):
                self.mu[j] = np.sum(gamma[:, j].reshape((-1, 1)) * x, axis=0) / np.sum(gamma[:, j])

                # 正则化, 避免出现奇异阵
                delta = 0.001 * np.identity(self.dim)
                self.cov[j] = np.dot((x - self.mu[j]).T, gamma[:, j].reshape((-1, 1)) * (x - self.mu[j])) / np.sum(gamma[:, j]) + delta

                self.alpha[j] = np.sum(gamma[:, j]) / N

            with open("train.txt", "a") as f:
                kl = gmm_kl(self, origin_model)
                js = gmm_js(self, origin_model)
                f.write(str(kl)+" "+str(js)+"\n")

        print("training is over!")
        print("mean:\n", self.mu)
        print("cov:\n", self.cov)
        print("alpha:\n", self.alpha)

    def score(self, x):
        scores = np.zeros([x.shape[0], self.k])
        for i in range(self.k):
            scores[:, i] = self.alpha[i] * multivariate_normal.pdf(x, self.mu[i], self.cov[i])
        return np.log(np.sum(scores, axis=1))

    def evaluate(self, x, y, mean, cov, data_scales, drawEllipse=True):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        colors = "ygb"
        for i in range(x.shape[0]):
            plt.plot(x[i][0], x[i][1], '.', color=colors[y[i][0]], markersize=2)

        #estimate
        for i in range(self.k):
            line = plt.plot(self.mu[i][0], self.mu[i][1], "x", color="red", markersize=15)
        plt.setp(line, label="GMM")

        #Compare with real dataset
        for i in range(len(mean)):
            line = plt.plot(mean[i][0], mean[i][1], "x", color="black", markersize=15)
        plt.setp(line, label="Origin Data")

        plt.legend()
        plt.show()

        if drawEllipse:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            for i in range(x.shape[0]):
                plt.plot(x[i][0], x[i][1], '.', color=colors[y[i][0]], markersize=2)

            # etimate
            for i in range(self.k):
                printEllipse(ax, self.mu[i], self.cov[i], color="red", shreshold="95")

            # real dataset
            for i in range(len(mean)):
                printEllipse(ax, mean[i], cov[i], color="black", shreshold="95")
            plt.show()

        fig = plt.figure()
        with open("train.txt", "r") as f:
            lines = f.readlines()
            epochs = len(lines)
            kls = [float(line.split()[0]) for line in lines]
            jss = [float(line.split()[1]) for line in lines]

        plt.plot(range(epochs), kls, label="KL")
        plt.plot(range(epochs), jss, label="JS")
        plt.legend()
        plt.xlabel("Epoch")
        plt.title("KL Divergence & JS Divergence")
        plt.show()


def gmm_kl(gmm1, gmm2):
    # 计算GMM与原始分布的KL散度
    # 采用蒙特卡洛方法做近似
    samples = gmm1.sample(1000)
    log_gmm1 = np.mean(gmm1.score(samples))
    log_gmm2 = np.mean(gmm2.score(samples))
    return log_gmm1 - log_gmm2

def gmm_js(gmm1,gmm2):
    # 对称且光滑的Jensen-Shannon散度
    samples1 = gmm1.sample(1000)
    log_gmm11 = np.mean(gmm1.score(samples1))
    log_gmm12 = np.mean(gmm2.score(samples1))

    samples2 = gmm2.sample(1000)
    log_gmm21 = np.mean(gmm1.score(samples2))
    log_gmm22 = np.mean(gmm2.score(samples2))

    return (np.mean(log_gmm11) - (np.mean(np.logaddexp(log_gmm11, log_gmm12)) - np.log(2))
            + np.mean(log_gmm22) - (np.mean(np.logaddexp(log_gmm21, log_gmm22)) - np.log(2))) / 2


if __name__  ==  "__main__":
    mean = [[-1, 0], [1, 1], [0, 2]]
    cov = [[[0.1, 0], [0, 0.1]], [[0.1, 0], [0, 0.1]], [[0.1, 0], [0, 0.1]]]
    data_scales = [100, 300, 500]
    createData(mean, cov, data_scales)

    x, y = loadData()

    model = GMM(x)
    model.train(x, epoch=20)
    model.evaluate(x, y, mean, cov, data_scales)

    origin_model = GMM(x, mean, cov, data_scales / np.sum(data_scales))
    kl = gmm_kl(model, origin_model)
    js = gmm_js(model, origin_model)
    print("KL散度(GMM与原始分布):{0}".format(kl))
    print("Jensen-Shannon散度(GMM与原始分布):{0}".format(js))



