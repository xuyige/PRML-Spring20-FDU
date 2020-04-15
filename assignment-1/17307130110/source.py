import numpy
from handout.code.data import *
from handout.code.model import LinearGenerativeModel, LinearDiscriminativeModel

def cal_accuracy(a, b):
    ''' calculate the accuracy between a and b

    Arguments:
        a: M * 1 matrix, standrad labels
        b: M * 1 matrix, checking labels

    Returns:
        accuracy: the accuracy between a and b
    '''
    M = a.shape[0]
    count = 0
    for i in range(M):
        if a[i, 0] == b[i, 0]:
            count += 1
    return count / M

test_times = 3

for i in range(test_times):
    print('test %d' % i)
    generate_data('handout/data/train.data', 1000)
    generate_data('handout/data/test.data', 1000)
    (train_M, train_samples, train_labels) = read_data('handout/data/train.data')
    (test_M, test_samples, test_labels) = read_data('handout/data/train.data')

    gen_model = LinearGenerativeModel()
    gen_model.train(train_samples, train_labels)
    gen_labels = gen_model.classify(test_samples)
    print('------Generative Model Accuracy: ', cal_accuracy(test_labels, gen_labels))

    dis_model = LinearDiscriminativeModel()
    dis_model.train(train_samples, train_labels, 0.8, 5, 5)
    dis_labels = dis_model.classify(test_samples)
    print('------Discriminative Model Accuracy: ', cal_accuracy(test_labels, dis_labels))
