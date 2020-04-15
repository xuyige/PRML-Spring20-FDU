import numpy as np
import matplotlib.pyplot as plt


def multivariate_normal(mean, cov, sample_num, expand_size=None):
    """
    generate sample_num data in single class in range (mean - expand_size , mean + expand_size)
    Note: if expand_size not specified, expand_size would be coded as Confidence Interval to hold 99% data
    :param mean: (variant_num)
    :param cov: (variant_num, variant_num)
    :param sample_num: (1, )
    :param expand_size: (1, )
    :return: (sample_num, variant_num)
    """
    if expand_size is None:
        expand_size = np.array(cov).diagonal() * 3
    sample = []
    if mean.shape[0] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
        print("Dimension error!")
        return None
    while len(sample) != sample_num:
        # norm_part = 1 / ((2 * np.pi) ** ((mean.shape[0] - 1) / 2) * (1/np.linalg.det(cov)) ** 1 / 2)
        u = (np.random.rand(mean.shape[0]) * 2 * expand_size - expand_size) + mean
        exp_part = -0.5 * np.mat(u - mean) * np.mat(cov) * np.mat(u - mean).T
        generator = np.random.rand(1)
        if np.exp(exp_part) >= generator:
            sample.append(u)
    return np.stack(sample, axis=0)


def normal_distribution_generate(mean, cov, class_num, sample_num):
    """
    create datasets with parameters
    :param mean: (class_num, variant_num)
    :param cov: (class_num, variant_num, variant_num)
    :param class_num: (1, )
    :param sample_num: (class_num, )
    :return: sample: (sample_num, variant_num + label_num)
    """
    sample_list = []
    for i in range(class_num):
        data_raw = multivariate_normal(mean[i, :], cov[i, :, :], sample_num[i]).T
        labels = np.ones_like(data_raw[0, :]) * i
        data = np.concatenate([data_raw, labels[None, :]], axis=0)
        sample_list.append(data)
    sample = np.concatenate(sample_list, axis=1)
    sample = np.transpose(sample, (1, 0))
    plt.plot(sample[:, 0], sample[:, 1], 'x')
    plt.show()
    return sample


def save_dataset(sample, file_name='data.data'):
    """
    Save the data and create dataset
    :param sample: (sample_num, variant_num + label)
    :param file_name: string
    :return: file with file_name
    """
    np.random.shuffle(sample)
    np.savetxt(file_name, sample, fmt='%f', delimiter=',')


def load_dataset(file_name):
    """
    laod dataset
    :param file_name:name of file
    :return: sample: (sample_num, variant_num + label)
    """
    sample = np.loadtxt(file_name, delimiter=',')
    return sample


# Least Square Classification
class LeastSquareDiscriminantModel:
    def __init__(self, file_name='data.data'):
        # data
        self.file_name = file_name
        self.data = load_dataset(self.file_name)
        self.x = self.data[:, :-1]                   # (sample_num, variant_num)
        self.y = self.data[:, -1]                    # (sample_num, 1)
        self.k = self.data.shape[1]                  # variant_num + 1
        self.n = self.data.shape[0]                  # sample_num
        self.class_num = int(self.y.max() + 1)       # class_num
        self.w = np.zeros((self.class_num, self.k))  # (class_num, variant_num + 1)

    def train(self, func='one-vs-other'):
        """
        Discriminate in two different ways
        :param func: "one-vs-other" or "one-vs-one"
        :return: None
        """
        w_list = []
        ones = np.ones(self.n)
        minus = ones * -1
        zeros = np.zeros(self.n)
        x = np.mat(np.concatenate((self.x, np.ones(self.n)[:, None]), axis=1))
        if func == 'one-vs-other':
            for i in range(self.class_num):
                t = np.where(self.y == i, ones, minus)
                w = np.resize(((x.T * x).I * x.T * np.mat(t).T), self.class_num)
                w_list.append(w)

        elif func == 'one-vs-one':
            for i in range(self.class_num):
                for j in range(i + 1, self.class_num):
                    t = np.where(self.y == i, ones, np.where(self.y == j, minus, zeros))
                    w = np.resize(((x.T * x).I * x.T * np.mat(t).T), self.class_num)
                    w_list.append(w)

        self.w = np.stack(w_list, axis=0)
        return self.w

    def display(self):
        """
        display the boundary with datasets in 2D
        :return: None
        """
        plt.plot(self.data[:, 0], self.data[:, 1], 'x')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        for i in range(self.w.shape[0]):
            x = [-10, 10]
            y = [0, 0]
            y[0] = (- self.w[i][2] - self.w[i][0] * x[0]) / self.w[i][1]
            y[1] = (- self.w[i][2] - self.w[i][0] * x[1]) / self.w[i][1]
            plt.plot(x, y)
        plt.title('Boundary for Least_Square')
        plt.show()


# Perceptron Classification
class PerceptronModel:
    def __init__(self, lr=0.01, error=0.001, file_name='data.data'):
        # data
        self.file_name = file_name
        self.data = load_dataset(self.file_name)
        self.x = self.data[:, :-1]                  # data:(sample_num, variant_num)
        self.y = self.data[:, -1]                   # label:(sample_num)
        self.k = self.data.shape[1]                 # variant_num + 1
        self.n = self.data.shape[0]                 # sample_num
        self.class_num = int(self.y.max() + 1)      # class_num

        # parameter
        self.w = np.zeros((self.class_num, self.k))
        self.lr = lr
        self.error = error

    def train(self):
        while True:
            old_w = self.w.copy()
            for i in range(self.n):
                for j in range(self.class_num):
                    if self.y[i] == j:
                        if np.dot(np.concatenate((self.x[i], np.ones(1))), self.w[j]) <= 0:
                            self.w[j, :] += self.lr * np.concatenate((self.x[i], np.ones(1)))
                    else:
                        if np.dot(np.concatenate((self.x[i], np.ones(1))), self.w[j]) > 0:
                            self.w[j, :] -= self.lr * np.concatenate((self.x[i], np.ones(1)))
            if (old_w == self.w).all():
                break
        return self.w

    def display(self):
        plt.plot(self.data[:, 0], self.data[:, 1], 'x')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        for i in range(self.w.shape[0]):
            x = [-10, 10]
            y = [0, 0]
            y[0] = (- self.w[i][2] - self.w[i][0] * x[0]) / self.w[i][1]
            y[1] = (- self.w[i][2] - self.w[i][0] * x[1]) / self.w[i][1]
            plt.plot(x, y)
        plt.title('Boundary for Perceptron')
        plt.show()


# Logistic_Discriminant_Classification
class LogisticDiscriminantModel:
    def __init__(self, lr=0.03, maxiter=200, error=0.0001, file_name='data.data'):
        # data
        self.file_name = file_name
        self.data = load_dataset(self.file_name)
        self.x = self.data[:, :-1]                          # (sample_num, variant_num)
        self.y = self.data[:, -1]                           # (sample_num)
        self.k = self.data.shape[1]                         # variant_num + 1
        self.n = self.data.shape[0]                         # sample_num
        self.class_num = int(self.y.max() + 1)              # class_num
        self.w = np.zeros((self.k - 1, self.class_num))     # (variant_num, class_num)
        self.b = np.zeros(self.class_num)                   # (class_num, )
        self.y_one_hot = []
        y_list = list(self.y)
        for num in y_list:
            self.y_one_hot.append([1 if i == num else 0 for i in range(self.class_num)])
        self.y_one_hot = np.stack(self.y_one_hot, axis=0)

        # learn
        self.lr = lr
        self.maxiter = maxiter
        self.error = error

    def train(self):
        """
        train the model with Gradient-Descent and call the display
        :return loss_record: (iter_times, )
        """
        loss_record = []
        iter = 0
        new_loss = self.cross_entropy(self.x, self.y_one_hot, self.w, self.b)
        print("loss before iteration: %f" % new_loss)
        while True:
            old_loss = new_loss
            iter += 1
            gradient_w, gradient_b = self.gradient(self.x, self.y_one_hot, self.w, self.b)
            self.w -= self.lr * gradient_w
            self.b -= self.lr * gradient_b
            new_loss = self.cross_entropy(self.x, self.y_one_hot, self.w, self.b)
            loss_record.append(new_loss)
            print("iteration %d loss %f" % (iter, new_loss))
            if abs(old_loss - new_loss) < self.error or iter > self.maxiter:
                break
        return loss_record

    def test(self):
        y_hat = self.softmax(np.dot(self.x, self.w) + self.b)
        one_hat = np.ones_like(y_hat)
        zero_hat = np.zeros_like(y_hat)
        y_list = []
        for i in range(y_hat.shape[0]):
            y = np.where(y_hat[i, :] == np.max(y_hat[i, :]), one_hat[0, :], zero_hat[0, :])
            y_list.append(y)
        y_hat_one_hot = np.stack(y_list)
        return 1.0 - np.where((self.y_one_hot - y_hat_one_hot) != 0, one_hat, zero_hat).sum() / y_hat.shape[0]

    def gradient(self, data, target, w, b):
        """
        calculate the gradient of mean & cov matrix
        :param data: (sample_num, variant_num)
        :param target: (sample_num, class_num)
        :param w: (variant_num, class_num)
        :param b: (class_num, )
        :return: gradient_w (variant_num, class_num), gradient_b (class_num, )
        """
        y_hat = self.softmax(np.dot(data, w) + b)
        gradient_w = -1 * np.mat(data.T) * np.mat(target - y_hat) / self.n
        gradient_b = (-1 * np.sum((target - y_hat), axis=0)) / self.n
        return gradient_w, gradient_b

    def cross_entropy(self, data, target, w, b):
        """
        compute the loss
        :param data: (sample_num, variant_num)
        :param target: (sample_num, class_num)
        :param w: (variant_num, class_num)
        :param b: (class_num, )
        :return loss: float
        """
        y_hat = self.softmax(np.dot(data, w) + b)
        loss = - np.trace(np.dot(target, np.log(y_hat).T)) / self.n
        return loss

    def softmax(self, x):
        """
        calculate the softmax
        :param x: (sample_num, class_num)
        :return: (sample_num, class_num)
        """
        x = x - np.mean(x, axis=1)[:, None].repeat(self.class_num, axis=1)
        return np.exp(x) / (np.sum(np.exp(x), axis=1))[:, None].repeat(self.class_num, axis=1)

    def display(self, loss_record):
        """
        display the visualization of classification boundary & loss_history
        :param loss_record: (iter_times, )
        :return: None
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title('Boundary for Discriminative Logistic')
        plt.plot(self.data[:, 0], self.data[:, 1], 'x')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        one_hat = np.array([1, 1, 1])
        zero_hat = np.array([0, 0, 0])
        for i in range(20):
            for j in range(20):
                data = np.array([i - 10, j - 10])
                y_hat = self.softmax(np.dot(data[None, :], self.w) + self.b).squeeze()
                y_hat = np.where(y_hat == np.max(y_hat), one_hat, zero_hat)
                if y_hat[0] == 1:
                    plt.plot(i - 10, j - 10, 'x', color='red')
                elif y_hat[1] == 1:
                    plt.plot(i - 10, j - 10, '.', color='green')
                else:
                    plt.plot(i - 10, j - 10, '+', color='blue')
        ax2 = fig.add_subplot(1, 2, 2)
        plt.plot(loss_record, linestyle='dashed')
        ax2.set_title('loss at iterations')
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('loss')
        plt.show()


# Logistic_Generative_Classification
class LogisticGenerativeModel:
    def __init__(self, lr=0.03, maxiter=200, error=0.00001, file_name='data.data'):
        # data
        self.file_name = file_name
        self.data = load_dataset(self.file_name)
        self.x = self.data[:, :-1]                          # (sample_num, variant_num)
        self.y = self.data[:, -1]                           # (sample_num)
        self.k = self.data.shape[1]                         # variant_num + 1
        self.n = self.data.shape[0]                         # sample_num
        self.class_num = int(self.y.max() + 1)              # class_num
        self.mean = np.zeros((self.class_num, self.k - 1))  # (class_num, variant_num)
        self.cov = np.identity(self.k - 1)                  # (variant_num, variant_num)
        self.pi = 1 / self.class_num
        self.y_one_hot = []
        y_list = list(self.y)
        for num in y_list:
            self.y_one_hot.append([1 if i == num else 0 for i in range(self.class_num)])
        self.y_one_hot = np.stack(self.y_one_hot, axis=0)

        # learn
        self.lr = lr
        self.maxiter = maxiter
        self.error = error

    def train(self):
        """
        train the model with Gradient-Descent and call the display
        :return loss_record: (iter_times, )
        """
        loss_record = []
        iter = 0
        new_loss = self.cross_entropy(self.x, self.y_one_hot, self.mean, self.cov)
        print("loss before iteration: %f" % new_loss)
        while True:
            old_loss = new_loss
            iter += 1
            gradient_cov, gradient_mean = self.gradient(self.x, self.y_one_hot, self.mean, self.cov)
            self.cov -= self.lr * gradient_cov
            self.mean -= self.lr * gradient_mean
            new_loss = self.cross_entropy(self.x, self.y_one_hot, self.mean, self.cov)
            loss_record.append(new_loss)
            print("iteration %d loss %f" % (iter, new_loss))
            if abs(old_loss - new_loss) < self.error or iter > self.maxiter:
                break
        return loss_record

    def test(self):
        y_hat = self.softmax(self.gaussian(self.x, self.mean, self.cov))
        one_hat = np.ones_like(y_hat)
        zero_hat = np.zeros_like(y_hat)
        y_list = []
        for i in range(y_hat.shape[0]):
            y = np.where(y_hat[i, :] == np.max(y_hat[i, :]), one_hat[0, :], zero_hat[0, :])
            y_list.append(y)
        y_hat_one_hot = np.stack(y_list)
        return 1.0 - np.where((self.y_one_hot - y_hat_one_hot) != 0, one_hat, zero_hat).sum() / y_hat.shape[0]

    def gradient(self, data, target, mean, cov):
        """
        calculate the gradient of mean & cov matrix
        :param data: (sample_num, variant_num)
        :param target: (sample_num, class_num)
        :param mean: (class_num, variant_num)
        :param cov: (variant_num, variant_num)
        :return:
        """
        y_hat = self.softmax(self.gaussian(data, mean, cov))
        identity = np.identity(self.class_num)
        gradient_cov = np.zeros_like(cov)
        for i in range(self.class_num):
            gradient_cov += np.mat(np.mat(mean[i, :][:, None])
                                   * np.mat(identity[i, :][:, None].T)
                                   * np.mat((target - y_hat)).T * np.mat(data)
                                   - 0.5 * np.mat(mean[i, :][:, None].repeat(self.n, axis=1))
                                   * np.mat((target - y_hat)) * np.mat(identity[i, :][:, None])
                                   * np.mat(mean[i, :][:, None]).T).T / self.n
        gradient_mean = 0
        mean_list = []
        for i in range(self.class_num):
            for j in range(self.n):
                gradient_mean += np.mat(np.mat(identity[i, :][:, None].T)
                                        * np.mat((target - y_hat)[j, :]).T
                                        * np.mat(data[j, :])
                                        * np.mat(cov)
                                        - 0.5 * np.mat(identity[i, :][:, None].T)
                                        * np.mat((target - y_hat)[j, :]).T
                                        * np.mat(mean[i, :][:, None]).T
                                        * np.mat(cov).T
                                        - 0.5 * np.mat((target - y_hat)[j, :])
                                        * np.mat(identity[i, :][:, None])
                                        * np.mat(mean[i, :][:, None]).T
                                        * np.mat(cov)).T / self.n
            mean_list.append(np.array(gradient_mean))
        gradient_mean = np.stack(mean_list, axis=0).squeeze()
        return gradient_cov, gradient_mean

    def cross_entropy(self, data, target, mean, cov):
        """
        compute the loss
        :param data: (sample_num, variant_num)
        :param target: (sample_num, class_num)
        :param mean: (class_num, variant_num)
        :param cov: (variant_num, variant_num)
        :return loss: float
        """
        y_hat = self.softmax(self.gaussian(data, mean, cov))
        loss = - np.trace(np.dot(target, np.log(y_hat).T)) / self.n
        return loss

    def softmax(self, x):
        """
        calculate the softmax
        :param x: (sample_num, class_num)
        :return: (sample_num, class_num)
        """
        x = x - np.mean(x, axis=1)[:, None].repeat(x.shape[1], axis=1)
        return np.exp(x) / (np.sum(np.exp(x), axis=1))[:, None].repeat(self.class_num, axis=1)

    def gaussian(self, x, mean, cov):
        """
        calculate the gaussian
        :param x: (sample_num, variant_num)
        :param mean: (class_num, variant_num)
        :param cov: (variant_num, variant_num)
        :return: (sample_num, class_num)
        """
        n_list = []
        for j in range(x.shape[0]):
            k_list = []
            for i in range(mean.shape[0]):
                norm_part = np.log(1 / ((2 * np.pi) ** ((x.shape[1] - 1) / 2) * (np.linalg.det(cov)) ** 1 / 2))
                vec = (x[j, :] - mean[i, :])
                exp_part = -0.5 * np.dot(np.dot(vec, cov), vec.T)
                k_list.append(norm_part * exp_part)
            n_list.append(np.stack(k_list, axis=0))
        return np.stack(n_list, axis=0)

    def display(self, loss_record):
        """
        display the visualization of classification boundary & loss_history
        :param loss_record: (iter_times, )
        :return: None
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title('Boundary for Generative Logistic')
        plt.plot(self.data[:, 0], self.data[:, 1], 'x')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        one_hat = np.array([1, 1, 1])
        zero_hat = np.array([0, 0, 0])
        for i in range(20):
            for j in range(20):
                data = np.array([i - 10, j - 10])
                y_hat = self.softmax(self.gaussian(data[None, :], self.mean[:, :2], self.cov[:2, :2])).squeeze()
                y_hat = np.where(y_hat == np.max(y_hat), one_hat, zero_hat)
                if y_hat[0] == 1:
                    plt.plot(i - 10, j - 10, 'x', color='red')
                elif y_hat[1] == 1:
                    plt.plot(i - 10, j - 10, '.', color='green')
                else:
                    plt.plot(i - 10, j - 10, '+', color='blue')
        ax2 = fig.add_subplot(1, 2, 2)
        plt.plot(loss_record, linestyle='dashed')
        ax2.set_title('loss at iterations')
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('loss')
        plt.show()
