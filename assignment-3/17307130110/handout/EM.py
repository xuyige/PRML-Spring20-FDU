import numpy as np

min_val = np.exp(-30)


def normal_probability_density(x, mean, cov):
    ''' 计算正态分布在某点的概率密度
    Arguments:
        x: 样本点
        mean: 均值
        cov: 协方差矩阵
    '''
    tmp = x.reshape(-1,1) - mean.reshape(-1,1)
    D = tmp.shape[0]

    r = 1.0 / np.sqrt(np.power(2 * np.pi, D)) * 1.0 / np.sqrt(np.linalg.det(cov)) * np.exp(-0.5 * np.dot(np.dot(tmp.T, np.linalg.inv(cov)),tmp))
    return r[0,0]


def kmeans(X, K, max_epoch=10000):
    ''' 实现kmeans，可以用于预计算均值
    Arguments:
        X: (N, D)的数据集，每行即一个样本点，D是维度
        K: 高斯分布个个数
        max_epoch: 最大迭代次数
    Returns:
        pi: K维向量，高斯分布的选取概率
        mean: (K, D)的矩阵，每一行是对应高斯分布的均值
        cov: (K, D, D)的矩阵,即协方差矩阵
        label: 标签
    '''
    N = X.shape[0]
    D = X.shape[1]
    pi = np.zeros(K)
    mean = np.zeros([K, D])
    cov = np.zeros([K, D, D])
    label = np.random.randint(0, K, N)
    dis = np.zeros(K)

    # 随机选取k个中心
    index = np.random.randint(0, N, K)
    for k in range(K):
        mean[k,:] = X[index[k],:]
    
    # 迭代进行kmeans
    epoch = -1
    while epoch < max_epoch:
        epoch += 1
        Change = False
        # print("Kmeans -- epoch %d -- label[0] %d, label[60] %d, label[90] %d" % (epoch, label[0], label[60], label[90]))
        # 分配步
        for n in range(N):
            for k in range(K):
                tmp = X[n,:] - mean[k,:]
                dis[k] = np.sqrt(np.sum(tmp**2))
            L = np.argmin(dis)
            if label[n] != L:
                label[n] = L
                Change = True
        # 更新步
        for k in range(K):
            index = [n for n in range(N) if label[n] == k]
            if index:
                tmp = X[index,:].reshape(-1,D)
                mean[k,:] = np.sum(tmp, axis=0) / tmp.shape[0]
        # 若稳定则终止
        if not Change:
            break
    
    # 计算pi
    for k in range(K):
        index = [n for n in range(N) if label[n] == k]
        pi[k] = len(index) / N
    
    # 计算cov, 仅考虑对角线
    for k in range(K):
        index = [n for n in range(N) if label[n] == k]
        tmp = X[index, :].reshape(-1, D)
        for i in range(len(index)):
            tmp[i,:] = tmp[i,:] - mean[k,:]
            tmp[i,:] = tmp[i,:] ** 2
        s = np.sum(tmp, axis=0) / len(index)
        for d in range(D):
            cov[k, d, d] = s[d]

    return pi, mean, cov, label
    

def simple_init(X, K):
    ''' 简单初始化，可用于EM的参数初始化
    '''
    N = X.shape[0]
    D = X.shape[1]
    pi = np.zeros(K)
    mean = np.zeros([K, D])
    cov = np.zeros([K, D, D])
    
    for k in range(K):
        pi[k] = 1.0 / K
    
    # 随机选取k个中心
    index = np.random.randint(0, N, K)
    for k in range(K):
        mean[k,:] = X[index[k],:]

    for k in range(K):
        for d in range(D):
            cov[k,d,d] = 1
    
    return pi, mean, cov


def myEM(X, K, max_epoch=10000, km_epoch=100):
    ''' 实现EM算法
    Arguments:
        X: (N, D)的数据集，每行即一个样本点，D是维度
        K: 高斯分布个个数
        max_epoch: 最大迭代次数
        km_epoch: 调用Kmeans初始化时，最大迭代次数
    Returns:
        pi: K维向量，高斯分布的选取概率
        mean: (K, D)的矩阵，每一行是对应高斯分布的均值
        cov: (K, D, D)的矩阵,即协方差矩阵
        label: 标签
    '''

    N = X.shape[0]
    D = X.shape[1]
    pi = np.zeros(K)
    mean = np.zeros([K, D])
    cov = np.zeros([K, D, D])
    gamma = np.zeros([N, K])
    
    # 初始化
    pi, mean, cov, tmp = kmeans(X, K, max_epoch=km_epoch)
    # pi, mean, cov = simple_init(X, K)

    epsilon = np.exp(-10)
    last_val = cur_val = -999999999
    epoch = 0

    while epoch < max_epoch:
        # E步：最小化KL距离，令变分分布等于p(z|x)，即计算gamma矩阵
        for n in range(N):
            for k in range(K):
                gamma[n,k] = pi[k] * normal_probability_density(X[n,:], mean[k,:], cov[k,:,:])
        s = np.sum(gamma, axis=1)
        # E步：归一化
        for n in range(N):
            if abs(s[n]) >= min_val:
                gamma[n,:] = gamma[n,:] / s[n]

        # M步：计算pi
        Nk = np.sum(gamma, axis=0)
        pi = Nk / N
       
        # M步：计算mean
        for k in range(K):
            mean[k,:] = np.zeros(D)
            for n in range(N):
                mean[k,:] += gamma[n, k] * X[n,:]
            if abs(Nk[k]) >= min_val:
                mean[k,:] /= Nk[k]
        
        # M步：计算cov
        for k in range(K):
            cov[k,:,:] = np.zeros([D, D])
            for n in range(N):
                tmp = (X[n,:] - mean[k,:]).reshape(-1,1)
                cov[k,:,:] += gamma[n, k] * np.dot(tmp, tmp.T)
            if abs(Nk[k]) >= min_val:
                cov[k,:,:] /= Nk[k]
        
        # 计算cur_val
        cur_val = 0
        s = np.sum(gamma, axis=1)
        for n in range(N):
            cur_val += np.log(s[n])

        # 判断Σlogp(x)是否收敛
        diff = abs(last_val - cur_val)
        print("EM - epoch %d, diff is %f, cur_val is %f" % (epoch, diff, cur_val))
        if diff < epsilon:
            break
        else:
            epoch += 1
            last_val = cur_val
    
    label = np.argmax(gamma, axis=1)

    return pi, mean, cov, label


    
