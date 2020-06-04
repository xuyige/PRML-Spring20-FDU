import numpy as np
import matplotlib.pyplot as plt


def generate_2D_data(N):
    ''' 生成混合二维高斯模型的样本数据
    Arguments:
        N: 样本点数
    Returns:
        K: 高斯分布个数
        samples: array-of-(N, D)，每一行即一个样本点
        labels: N维数组，表示对应样本点的标签
    '''
    K = 3

    freq = [0.7, 0.2, 0.1]

    mean0 = [50, 50]
    cov0 = [[35, 10], [10, 35]]
    mean1 = [70, 50]
    cov1 = [[10, 5], [5, 10]]
    mean2 = [50, 70]
    cov2 = [[10, 5], [5, 10]]
    
    s0 = np.random.multivariate_normal(mean0, cov0, int(N * freq[0]))
    s1 = np.random.multivariate_normal(mean1, cov1, int(N * freq[1]))
    s2 = np.random.multivariate_normal(mean2, cov2, int(N * freq[2]))

    samples = np.concatenate((s0, s1, s2), axis=0)
    labels = [0] * int(N * freq[0]) + [1] * int(N * freq[1]) + [2] * int(N * freq[2])
    
    return K, samples, labels


def generate_3D_data(N):
    ''' 生成混合三维高斯模型的样本数据
    Arguments:
        N: 样本点数
    Returns:
        samples: array-of-(N, D)，每一行即一个样本点
    '''
    K = 3

    freq = [0.5, 0.3, 0.2]

    mean0 = [50, 50, 50]
    cov0 = [[50, 5, 10],
            [5, 50, 5],
            [10, 5, 50]]

    mean1 = [100, 50, 40]
    cov1 = [[30, 5, 5],
            [5, 30, 5],
            [5, 5, 30]]

    mean2 = [50, 100, 100]
    cov2 = [[40, 5, 15],
            [5, 40, 5],
            [15, 5, 40]]
    
    s0 = np.random.multivariate_normal(mean0, cov0, int(N * freq[0]))
    s1 = np.random.multivariate_normal(mean1, cov1, int(N * freq[1]))
    s2 = np.random.multivariate_normal(mean2, cov2, int(N * freq[2]))

    samples = np.concatenate((s0, s1, s2), axis=0)
    labels = [0] * int(N * freq[0]) + [1] * int(N * freq[1]) + [2] * int(N * freq[2])
    
    return K, samples, labels


def draw_2D_data(samples, labels, no, title):
    ''' 对二维数据，绘制可视化散点图, 最大仅限4类数据
    Arguments:
        samples: (N, 2)的数据集
        labels: 标签
        np: 编号
        title: 标题
    '''
    plt.figure(no)
    plt.title(title)

    N = samples.shape[0]
    colors = ['b', 'g', 'r', 'orange']
    
    for i in range(N):
        X = np.asarray(samples[i,0])
        Y = np.asarray(samples[i,1])
        plt.scatter(X, Y, c=colors[labels[i]], s=10)
    
