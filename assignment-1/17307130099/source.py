import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

#三个类分布的参数
mean1 = (2, 5)
cov1 = [[1, 0], [0, 1]]
mean2 = (5, 2)
cov2 = [[1, 0], [0, 1]]
mean3 = (6, 6)
cov3 = [[1, 0], [0, 1]]

#生成训练数据
def TrainData():
    dist_train1 = np.random.multivariate_normal(mean1, cov1, 1000)
    dist_train2 = np.random.multivariate_normal(mean2, cov2, 1000)
    dist_train3 = np.random.multivariate_normal(mean3, cov3, 1000)
    out1 = dist_train1.transpose()
    out2 = dist_train2.transpose()
    out3 = dist_train3.transpose()
    plt.plot(out1[0], out1[1], 'bo')
    plt.plot(out2[0], out2[1], 'ro')
    plt.plot(out3[0], out3[1], 'yo')
    #plt.savefig('./train.png')
    plt.show()

    #将生成的数据以字符串形式写入数据文件
    data = []
    num = len(dist_train1) + len(dist_train2) + len(dist_train3)
    for i in dist_train1:
        data.append(str(i[0]) + ' ' + str(i[1]) + ' ' + '0' + '\n')
    for i in dist_train2:
        data.append(str(i[0]) + ' ' + str(i[1]) + ' ' + '1' + '\n')
    for i in dist_train3:
        data.append(str(i[0]) + ' ' + str(i[1]) + ' ' + '2' + '\n')
    np.random.shuffle(data)
    fw = open('data.data', 'w')
    fw.write(str(num) + '\n')
    for i in data:
        fw.write(i)
    fw.close()

#生成测试数据
def TestData():
    dist_test1 = np.random.multivariate_normal(mean1, cov1, 500)
    dist_test2 = np.random.multivariate_normal(mean2, cov2, 500)
    dist_test3 = np.random.multivariate_normal(mean3, cov3, 500)
    out1 = dist_test1.transpose()
    out2 = dist_test2.transpose()
    out3 = dist_test3.transpose()
    plt.plot(out1[0], out1[1], 'bo')
    plt.plot(out2[0], out2[1], 'ro')
    plt.plot(out3[0], out3[1], 'yo')
    #plt.savefig('./test.png')
    plt.show()

    # 将生成的数据以字符串形式写入数据文件
    data = []
    num = len(dist_test1) + len(dist_test2) + len(dist_test3)
    for i in dist_test1:
        data.append(str(i[0]) + ' ' + str(i[1]) + ' ' + '0' + '\n')
    for i in dist_test2:
        data.append(str(i[0]) + ' ' + str(i[1]) + ' ' + '1' + '\n')
    for i in dist_test3:
        data.append(str(i[0]) + ' ' + str(i[1]) + ' ' + '2' + '\n')
    np.random.shuffle(data)
    fw = open('test.data', 'w')
    fw.write(str(num) + '\n')
    for i in data:
        fw.write(i)
    fw.close()

#读入数据文件中的数据
def InputData(file_name):
    fr = open(file_name, 'r')
    lines_num = int(fr.readline())
    x = np.empty((lines_num, 2))
    t = np.empty((lines_num, 1))
    for i in range(lines_num):
        (x[i][0], x[i][1], t[i]) = fr.readline().split()
    t = t.astype(int)
    fr.close()
    return lines_num, x, t

#生成模型训练函数
def Gentraining(n, x, t):
    class_num = np.zeros((3, 1))   #用于统计各类的数据个数
    pi = np.zeros((3, 1))          #各类的先验概率
    mu = np.zeros((3, 2))
    sigma = np.zeros((2, 2))

    for i in range(n):            #统计各类数据个数并判断数据是否合规
        if 0 <= t[i] < 3:         #生成模型中类标签使用标量值更加方便
            class_num[t[i]] += 1
            mu[t[i]] += x[i]
        else:
            print("数据点(" + str(x[i][0]) + "," + str(x[i][1]) + "),类别不在目标范围")

    #依照公式求各参数
    for i in range(3):
        pi[i] = class_num[i] / n
        mu[i] = mu[i] / class_num[i]

    for i in range(n):
        sigma += np.dot((x[i] - mu[t[i]]).transpose(), x[i] - mu[t[i]])
    sigma /= n

    return pi, mu, sigma

#生成模型决策函数
def GenClassify(x, pi, mu, sigma):
    p_x_C = np.empty((3, 1))
    a = np.empty((3, 1))
    for i in range(3):
        p_x_C[i] = st.multivariate_normal.pdf(x, mean=mu[i], cov=sigma) #计算样本x相对各类的条件概率
        a[i] = np.log(p_x_C[i] * pi[i])
    #最大的a[i]就对应最大的后验概率
    return np.argmax(a)

#生成模型测试函数
def GenTest():
    n_train, x_train, t_train = InputData('data.data')
    pi, mu, sigma = Gentraining(n_train, x_train, t_train)
    n_test, x_test, t_test = InputData('test.data')
    correct = 0
    for i in range(n_test):
        if t_test[i] == GenClassify(x_test[i], pi, mu, sigma):
            correct += 1
    print('generative model:\ncorrect/total')
    print(correct, '/', n_test)
    print('accuracy rate:', correct/n_test)
    return

#判别模型
#softmax函数
def softmax(w, x):
    nu = 0
    y = np.zeros(3)
    for i in range(3):
        nu += np.exp(np.dot(w[i], x))
    for i in range(3):
        y[i] = np.exp(np.dot(w[i], x)) / nu
    #print(y)
    return y

#由于判别模型使用增广特征向量将原有二维数据转换为三维数据
def TransX(x, n):
    trans_x = np.empty((n, 3))
    for i in range(n):
        trans_x[i][0] = x[i][0]
        trans_x[i][1] = x[i][1]
        trans_x[i][2] = 1
    return trans_x

#判别模型训练函数
def DisTraining(n, x, t):
    w = np.zeros((3, 3))
    alpha = 1
    t_vector = np.empty((n, 3))
    for i in range(n):                #判别模型类标签使用向量更方便转换原有类标签
        if t[i] == 0:
            t_vector[i] = np.array([1, 0, 0])
        elif t[i] == 1:
            t_vector[i] = np.array([0, 1, 0])
        elif t[i] == 2:
            t_vector[i] = np.array([0, 0, 1])
        else:
            print("数据点(" + str(x[i][0]) + "," + str(x[i][1]) + "),类别不在目标范围")
    for i in range(1000):
        sum = np.zeros((3, 3))
        for j in range(n):
            y = softmax(w, x[j])
            for k in range(3):
                for m in range(3):
                    sum[k][m] += x[j][k]*(t_vector[j][m]-y[m])   #用一个两重循环计算梯度公式中的向量相乘
        w = w + alpha * sum.transpose() / n
    return w

#判别模型决策函数
def DisClassify(x, w):
    y = softmax(w, x)
    return np.argmax(y)

#判别模型测试函数
def DisTest():
    n_train, x_train, t_train = InputData('data.data')
    x_train_trans = TransX(x_train, n_train)
    w = DisTraining(n_train, x_train_trans, t_train)
    n_test, x_test, t_test = InputData('test.data')
    x_test_trans = TransX(x_test, n_test)
    correct = 0
    for i in range(n_test):
        if t_test[i] == DisClassify(x_test_trans[i], w):
            correct += 1
    print('discriminative model:\ncorrect/total')
    print(correct, '/', n_test)
    print('accuracy rate:', correct / n_test)
    return

TrainData()
TestData()
GenTest()
DisTest()