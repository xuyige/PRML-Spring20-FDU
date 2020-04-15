import numpy as np
import argparse
import time
import matplotlib.pyplot as plt


d = 3
n_test = 1
pr = [0.333, 0.333, 0.334]

def generate_data(n, d, mean, cov, pr, train_p=None, file_path='./', n_test=1):
    '''

    :param n:
    :param d:
    :param mean:
    :param cov:
    :param pr:
    :param train_p:
    :param file_path:
    :return:
    '''
    if train_p is None:
        train_p = [.6, .2, .2]
    mean, cov, pr = np.asarray(mean), np.asarray(cov), np.asarray(pr)
    data, label = np.zeros((n, d)), np.zeros(n, dtype=int)
    sample_dist = (n * pr[:]).astype(int)
    sample_dist[-1] = n - np.sum( (n * pr[:-1]).astype(int))
    for i, ni in enumerate(sample_dist):
        it, end = np.sum(sample_dist[:i]), np.sum(sample_dist[:i +1])
        label[it:end], data[it:end] = i * np.ones(end - it, dtype=int), np.random.multivariate_normal(mean[i], cov[i], (ni,))

    index = np.arange(n)
    np.random.shuffle(index)
    label, data = label[index,], data[index,]
    train_n, test_n, valid_n = int(n * train_p[0]), int(n * train_p[1]), int(n * train_p[2])

    # train data
    df = ""
    for i in range(train_n):
        for j in range(data.shape[1]):
            df += str(data[i, j]) + ','
        df += str(label[i]) +'\n'
    fp = open(file_path + 'train_data_' + str(n_test) + '.data', 'w')
    fp.write(df)
    fp.close()

    # test data
    df = ""
    for i in range(train_n, test_n + train_n):
        for j in range(data.shape[1]):
            df += str(data[i, j]) + ','
        df += str(label[i]) +'\n'
    fp = open(file_path + 'test_data_' + str(n_test) +  '.data', 'w')
    fp.write(df)
    fp.close()

    # valid data
    df = ""
    for i in range(train_n + test_n, n):
        for j in range(data.shape[1]):
            df += str(data[i, j]) + ','
        df += str(label[i]) + '\n'
    fp = open(file_path + 'valid_data_' + str(n_test) + '.data', 'w')
    fp.write(df)
    fp.close()


def load_data(file_path='./train_data.data'):
    '''

    :param file_path:
    :return:
    '''
    fp = open(file_path)
    data_set = fp.read()
    data_set = data_set.split('\n')
    labels = []
    data = []
    for lines in data_set[:-1]:
        lines = lines.split(',')
        val = [float(v) for v in lines[:-1]]
        data.append(val)
        labels.append(int(lines[-1]))

    data = np.asarray(data)
    labels = np.asarray(labels)
    return data, labels


def plot_3d_data(x, labels, save_path='./data_0413'):
    '''

    :param x:
    :param labels:
    :param save_path:
    :return:
    '''
    x_A, y_A, z_A = [], [], []
    x_B, y_B, z_B = [], [], []
    x_C, y_C, z_C = [], [], []
    for i in range(len(x)):
        if labels[i] == 0:
            x_A.append(x[i, 0])
            y_A.append(x[i, 1])
            z_A.append(x[i, 2])
        if labels[i] == 1:
            x_B.append(x[i, 0])
            y_B.append(x[i, 1])
            z_B.append(x[i, 2])
        if labels[i] == 2:
            x_C.append(x[i, 0])
            y_C.append(x[i, 1])
            z_C.append(x[i, 2])
    ax = plt.axes(projection='3d')
    ax.scatter(x_A, y_A, z_A, c='y')
    ax.scatter(x_B, y_B, z_B, c='r')
    ax.scatter(x_C, y_C, z_C, c='g')
    ax.set_zlabel('z')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.savefig(save_path + '_' +str(time.localtime().tm_hour) + '_' + str(time.localtime().tm_min) + '.png')
    plt.show()


class generative_model:
    def __init__(self):
        pass

    def train_classifier(self, x, label):
        self.n = len(x)
        self.d = len(x[0])
        self.x = x
        self.label = label
        self.label_set = list(set(self.label))
        self.c = len(self.label_set)
        self.mean, self.cov = [], np.zeros((self.d, self.d))
        self.cov = np.zeros((self.c, self.d, self.d))
        self.num = [ sum(label == i) for i in range(self.c)]
        self.w = np.asmatrix(np.zeros((self.d, self.c)))
        self.b = np.zeros(self.c)

        for i in range(self.c):
            temp_n = np.asarray([data for (j, data) in enumerate(x) if label[j] == i])
            temp_mean = np.asarray([np.mean(temp_n[:, j]) for j in range(self.d)])
            temp_cov = np.asmatrix(np.cov(np.transpose(temp_n)))
            self.mean.append(temp_mean)
            self.cov += self.num[i] / self.n * temp_cov
            self.w[:, i] = (np.dot(temp_cov.I, temp_mean)).transpose()
            self.b[i] = - 1 / 2 * np.dot((temp_mean.transpose()).dot(temp_cov.I), temp_mean) + np.log(self.num[i] / self.n)



    def test_classifier(self, x, label, dataset):
        y_predict = np.asarray([self.classify_data(data) for data in x])
        accuracy = sum(y_predict == label) / len(label)
        recall = [np.sum(np.asarray([y_predict[j] == i and label[j] == i for j in range(len(x))])) / np.sum(label == i)for i in range(self.c)]

        print('[%s]generative model:\n Accuracy: %.4f'%(dataset, accuracy))
        for i in range(self.c):
            print('Recall of class %d: %.4f'%(i, recall[i]))



    def classify_data(self, x):
        alpha = self.w.transpose().dot(x.transpose()) + self.b
        alpha = np.asarray(alpha)
        y = np.exp(alpha) / np.sum(np.exp(alpha))
        return np.argmax(y)


class discriminative_model:
    def __init__(self):
        pass

    def train_classifier(self, x, label, n_c, n_iter = 8000, lr = 0.02):
        self.n = len(x)
        self.dim_n = len(x[0])
        self.class_n = n_c
        self.w = np.random.rand(self.class_n, self.dim_n)
        self.b = np.zeros((1, self.class_n))
        losses = []
        self.y = np.zeros((self.n, self.class_n))
        for i in range(self.n):
            self.y[i][label[i]] = 1


        for i in range(n_iter):
            val = np.dot(x, self.w.transpose()) + self.b
            p = np.exp(val) / np.sum(np.exp(val), axis=1, keepdims=True)
            loss = self.compute_cross_entropy(self.y, p)
            losses.append(loss)

            dw = (1 / self.n) * np.dot(x.transpose(), (p - self.y))
            db = (1 / self.n) * np.sum(p - self.y, axis=0)

            self.w = self.w - lr * dw.transpose()
            self.b = self.b - lr * db.transpose()

            if i % 1000 == 0:
                print('Iteration:[%d], loss:[%.4f]'%(i, loss))


    def compute_cross_entropy(self, y, val):
        return -(1 / self.n) * np.sum(y * np.log(val))


    def test_classifier(self, x, label, dataset):
        n_test = len(x)
        y_predict = np.zeros(n_test)
        for i in range(n_test):
            val = np.dot(x[i], self.w.transpose()) + self.b
            p = np.exp(val) / np.sum(np.exp(val), axis=1, keepdims=True)
            y_predict[i] = np.argmax(p, axis = 1)

        accuracy = sum(y_predict == label) / len(label)
        recall = [np.sum(np.asarray([y_predict[j] == i and label[j] == i for j in range(n_test)])) / np.sum(label == i) for i in range(self.class_n)]
        print('[%s]discriminative model: Accuracy: %.4f' % (dataset, accuracy))
        for i in range(self.class_n):
            print('Recall of class %d: %.4f' % (i, recall[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int , default=1000)
    parser.add_argument('--mean', type=str, default='-1, -1, -0, 1, 1, 3, 2, 4, 2')
    parser.add_argument('--cov', type=str, default='1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 3, 2, 0, 0, 0, 1, 0, 0, 0, 1')
    parser.add_argument('--iter', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.05)
    args = parser.parse_args()
    n, n_iter, lr = args.n, args.iter, args.lr
    mean_str, cov_str = args.mean, args.cov
    mean , cov = [], []
    try:
        mean_str = mean_str.split(',')
        for i in range(3):
            mean.append([float(mean_str[i * d + j]) for j in range(d)])
    except:
        print('the input of args mean is wrong ')
    try:
        cov_str = cov_str.split(',')
        for i in range(3):
            cov_temp = []
            for j in range(d):
                cov_temp.append([float(cov_str[i * d * d + j * d + k]) for k in range(d)])
            cov.append(cov_temp)
    except:
        print('the input of args cov is wrong ')

    generate_data(n, d, mean, cov, pr, n_test=n_test)
    data, labels = load_data('./train_data_'+ str(n_test)+ '.data')
    valid_data, valid_label = load_data('./valid_data_'+ str(n_test)+ '.data')
    test_x, test_y = load_data('./test_data_'+ str(n_test)+ '.data')
    if d  == 3:
        plot_3d_data(data, labels)

    generative_model = generative_model()
    generative_model.train_classifier(data, labels)
    generative_model.test_classifier(valid_data, valid_label, 'valid data')
    generative_model.test_classifier(test_x, test_y, 'test data')

    discriminative_model = discriminative_model()
    discriminative_model.train_classifier(data, labels, 3, n_iter, lr)
    discriminative_model.test_classifier(valid_data, valid_label, 'valid data')
    discriminative_model.test_classifier(test_x, test_y, 'test data')