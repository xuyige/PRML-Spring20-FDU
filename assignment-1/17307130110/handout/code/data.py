import random
import numpy


def generate_data(filename, num):
    ''' Use 3 two-dimensional normal distributions to generate data in the format of (x1, x2, y).
        Note y is the label. (x1, x2) is a random point.
        And write the dataset into a file
    '''
    mean1 = [10,10]
    cov1 = [[1,0],[0,1]]
    mean2 = [0,0]
    cov2 = [[1,0],[0,1]]
    mean3 = [-10,-10]
    cov3 = [[1,0],[0,1]]
    
    dataset = []
    dataset1 = numpy.random.multivariate_normal(mean1, cov1, num)
    dataset2 = numpy.random.multivariate_normal(mean2, cov2, num)
    dataset3 = numpy.random.multivariate_normal(mean3, cov3, num)
    for element in dataset1:
        dataset.append((element[0], element[1], 0))
    for element in dataset2:
        dataset.append((element[0], element[1], 1))
    for element in dataset3:
        dataset.append((element[0], element[1], 2))
    random.shuffle(dataset)
    
    with open(file=filename, mode='w', encoding='utf-8') as data_file:
        data_file.write(str(len(dataset)) + '\n')
        for element in dataset:
            data_file.write(str(element[0])+','+str(element[1])+','+str(element[2])+'\n')
 
 
def read_data(filename):
    ''' read data from the data file
    
    Returns:
        M: the number of samples
        X: M * 2 martix, 2-D coordinates
        Y: M * 1 martix, label values
    '''
    data_file = open(file=filename, mode='r', encoding='utf-8')
    N = int(data_file.readline())
    X = numpy.asmatrix(numpy.zeros([N, 2], dtype=numpy.float))
    Y = numpy.asmatrix(numpy.zeros([N, 1], dtype=numpy.int))
    for i in range(N):
        tmp = data_file.readline().split(',')
        (X[i, 0], X[i, 1], Y[i, 0]) = tmp
    data_file.close()
    return (N, X, Y)

