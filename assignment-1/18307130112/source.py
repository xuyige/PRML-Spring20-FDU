import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# Construct three sets of gaussian distribution data
def datacreate():
    mean1, mean2, mean3, cov, size1, size2, size3 = (0, 0), (4, 4), (2, 3), [[1, 0], [0, 1]], 128, 128, 8
    # Data is written to .data file
    datafile = open("data.data", "w")
    datafile.write("%s\n" % (str((size1 + size2 + size3) * 2)))
    plt.figure(figsize=(6, 6))
    # The training set
    gaus1 = np.random.multivariate_normal(mean1, cov, size1)
    gaus2 = np.random.multivariate_normal(mean2, cov, size2)
    gaus3 = np.random.multivariate_normal(mean3, cov, size3)
    for i in range(size1):
        datafile.write("%s %s %s\n" % (str(gaus1[i][0]), str(gaus1[i][1]), 0))
    for i in range(size2):
        datafile.write("%s %s %s\n" % (str(gaus2[i][0]), str(gaus2[i][1]), 1))
    for i in range(size3):
        datafile.write("%s %s %s\n" % (str(gaus3[i][0]), str(gaus3[i][1]), 2))
    # Construct the train data corresponding to the graph
    gaus_output1 = gaus1.transpose()
    gaus_output2 = gaus2.transpose()
    gaus_output3 = gaus3.transpose()
    plt.plot(gaus_output1[0], gaus_output1[1], 'ro')
    plt.plot(gaus_output2[0], gaus_output2[1], 'bo')
    plt.plot(gaus_output3[0], gaus_output3[1], 'go')

    # The testing set
    gaus1 = np.random.multivariate_normal(mean1, cov, size1)
    gaus2 = np.random.multivariate_normal(mean2, cov, size2)
    gaus3 = np.random.multivariate_normal(mean3, cov, size3)
    for i in range(size1):
        datafile.write("%s %s %s\n" % (str(gaus1[i][0]), str(gaus1[i][1]), 0))
    for i in range(size2):
        datafile.write("%s %s %s\n" % (str(gaus2[i][0]), str(gaus2[i][1]), 1))
    for i in range(size3):
        datafile.write("%s %s %s\n" % (str(gaus3[i][0]), str(gaus3[i][1]), 2))
    # Construct the test data corresponding to the graph
    gaus_output1 = gaus1.transpose()
    gaus_output2 = gaus2.transpose()
    gaus_output3 = gaus3.transpose()
    plt.plot(gaus_output1[0], gaus_output1[1], 'rs')
    plt.plot(gaus_output2[0], gaus_output2[1], 'bs')
    plt.plot(gaus_output3[0], gaus_output3[1], 'gs')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("The Chart of Three Gaussian Distribution")
    plt.show()
    datafile.close()
# Read data from data.data file
def readdata():
    datafile = open("data.data", "r")
    size = int(datafile.readline())
    data = np.empty([size, 2], float)
    type = np.empty([size, 1], int)
    for i in range(size):
        (data[i][0], data[i][1], type[i]) = datafile.readline().split()
        #print(data[i][0], data[i][1], type[i])
    datafile.close()
    return (size, data, type)

# Generative model building
# Getting training set`s probability, mean and covariance
# size = ex_size/2, data = ex_Data[0:size/2]
def init_generative(size, data, type):
    prob = np.zeros(3)
    mean = np.zeros([3, 2])
    cov  = np.zeros([2, 2])
    for i in range(size):
        prob[type[i]] += 1
    sizes = prob.astype(int)
    mean[0] = np.mean(data[0 : sizes[0]], axis=0)
    mean[1] = np.mean(data[sizes[0] : sizes[0]+sizes[1]], axis=0)
    mean[2] = np.mean(data[sizes[0]+sizes[1] : sizes[0]+sizes[1]+sizes[2]], axis=0)
    prob /= size
    for i in range(size):
        cov += np.dot((data[i] - mean[type[i]]).transpose(), data[i] - mean[type[i]])
    cov /= size
    return(prob, mean, cov, sizes[0], sizes[1], sizes[2])
# Estimate indicator labels for test data
def maxitem(p):
    if p[0] > p[1] and p[0] > p[2]:
        return 0
    elif p[2] > p[1] and p[2] > p[0]:
        return 2
    elif p[1] > p[2] and p[1] > p[0]:
        return 1
# Generate the prediction data corresponding to the test data
# size = ex_size/2, data = ex_Data[size/2:size]
def predict_generative(size, data, prob, mean, cov):
    pred = np.empty(size)
    for k in range(size):
        p = np.empty(3)
        sum = 0.0
        for i in range(3):
            p[i] = multivariate_normal(mean[i], cov).pdf(data[k]) * prob[i]
            sum += p[i]
        for i in range(3):
            p[i] /= sum
        pred[k] = maxitem(p)
    return (pred)
# Verify the test data and graph it
def test_generative(size, data, type, pred):
    ans = 0
    seq = int(size/2)
    ierr1, ierr2, ierr3 = 0, 0, 0
    for i in range(seq):
        if type[i + seq] == pred[i]:
            ans += 1
            if type[i + seq] == 0:
                ierr1 += 1
            if type[i + seq] == 1:
                ierr2 += 1
            if type[i + seq] == 2:
                ierr3 += 1
    print(ierr1, ierr2, ierr3)
    print("Generative model: the number of correct predictions is %s / %s = %s \n"%(str(ans), str(seq), str(ans/seq)))
    trainmode = ['rs', 'bs', 'gs']
    rightmode = ['ro', 'bo', 'go']
    wrongmode = ['rx', 'bx', 'gx']
    datas = data.transpose()

    for i in range(seq):
        plt.plot(datas[0][i], datas[1][i], trainmode[type[i][0]])
    # print("Right data:\n")
    for i in range(seq):
        if type[i + seq] == pred[i]:
            # print(data[i + seq], (" %d \n" % (type[i + seq] + 1)))
            plt.plot(data[i + seq][0], data[i + seq][1], rightmode[type[i + seq][0]])
    print("Wrong data:\n")
    for i in range(seq):
        if type[i + seq] != pred[i]:
            #print(data[i + seq], "real: %d predict: %d \n"%(type[i + seq][0], int(pred[i])))
            plt.plot(data[i + seq][0], data[i + seq][1], wrongmode[type[i + seq][0]])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("The Chart of Generative Mode")
    plt.show()

# Fisher Discriminative model building (LDA)
# Getting train data set`s mean, inner-class distance and w, y ( y = w.transpose()*x )
def init_discriminative(a, b):
    mean1, mean2 = np.mean(a, axis=0), np.mean(b, axis=0)
    A, B = a - mean1, b - mean2
    n1, n2= A.shape[0], B.shape[0]
    inner1 = inner2 = 0
    for i in range(n1):
        inner1 += np.dot(A[i:i+1,:].transpose(), A[i:i+1,:])
    for i in range(n2):
        inner2 += np.dot(B[i:i+1,:].transpose(), B[i:i+1,:])

    Sw = inner1 + inner2
    Sw_I = np.linalg.inv(Sw)
    w = np.dot(Sw_I, mean1 - mean2)
    y1 = np.dot(mean1, w.transpose())
    y2 = np.dot(mean2, w.transpose())
    return (w.transpose(), y1, y2)

# Estimate indicator labels for test data
# Generate the prediction data corresponding to the test data
def minitem(p):
    if p[0] < p[1] and p[0] < p[2]:
        return 0
    elif p[2] < p[1] and p[2] < p[0]:
        return 2
    elif p[1] < p[2] and p[1] < p[0]:
        return 1
def predict_discriminative_one_against_the_rest(size, tra, trb, trc, te):
    pred = np.empty(size)
    w1_t, y1a, y1bc = init_discriminative(tra, np.r_[trb, trc])
    w2_t, y2b, y2ac = init_discriminative(trb, np.r_[tra, trc])
    w3_t, y3c, y3ab = init_discriminative(trc, np.r_[tra, trb])
    for k in range(size):
        a, b, c = np.dot(te[k], w1_t), np.dot(te[k], w2_t), np.dot(te[k], w3_t)
        if (abs(y1a - a) <= abs(y1bc - a)) and (abs(y2ac - b) <= abs(y2b - b)) and (abs(y3ab - c) <= abs(y3c - c)):
            pred[k] = 0
        elif (abs(y1bc - a) <= abs(y1a - a)) and (abs(y2b - b) <= abs(y2ac - b)) and (abs(y3ab - c) <= abs(y3c - c)):
            pred[k] = 1
        elif (abs(y1bc - a) <= abs(y1a - a)) and (abs(y2ac - b) <= abs(y2b - b)) and (abs(y3c - c) <= abs(y3ab - c)):
            pred[k] = 2
        else:
            pred[k] = 3
    return (pred)

def predict_discriminative_pairwise_classification(size, tra, trb, trc, te):
    pred = np.empty(size)
    w1_t, y1a, y1b = init_discriminative(tra, trb)
    w2_t, y2a, y2c = init_discriminative(tra, trc)
    w3_t, y3b, y3c = init_discriminative(trb, trc)
    for k in range(size):
        a, b, c = np.dot(te[k], w1_t), np.dot(te[k], w2_t), np.dot(te[k], w3_t)
        if (abs(y1a - a) <= abs(y1b - a)) and (abs(y2a - b) <= abs(y2c - b)):
            pred[k] = 0
        elif (abs(y1b - a) <= abs(y1a - a)) and (abs(y3b - c) <= abs(y3c - c)):
            pred[k] = 1
        elif (abs(y2c - b) <= abs(y2a - b)) and (abs(y3c - c) <= abs(y3b - c)):
            pred[k] = 2
    return (pred)

# Verify the testing data and graph it
def test_discriminative(size, data, type, pred, s):
    ans = 0
    seq = int(size/2)
    ierr1, ierr2, ierr3 = 0, 0, 0
    for i in range(seq):
        if type[i + seq] == pred[i]:
            ans += 1
            if type[i + seq] == 0:
                ierr1 += 1
            if type[i + seq] == 1:
                ierr2 += 1
            if type[i + seq] == 2:
                ierr3 += 1
    print(ierr1, ierr2, ierr3)

    print("Discriminative model: the number of correct predictions is %s / %s = %s \n"%(str(ans), str(seq), str(ans/seq)))
    trainmode = ['rs', 'bs', 'gs']
    rightmode = ['ro', 'bo', 'go']
    wrongmode = ['rx', 'bx', 'gx']
    datas = data.transpose()
    for i in range(seq):
        plt.plot(datas[0][i], datas[1][i], trainmode[type[i][0]])
    # print("Right data:\n")
    for i in range(seq):
        if type[i + seq] == pred[i]:
            # print(data[i + seq], (" %d \n" % (type[i + seq] + 1)))
            plt.plot(data[i + seq][0], data[i + seq][1], rightmode[type[i + seq][0]])
    print("Wrong data:\n")
    for i in range(seq):
        if type[i + seq] != pred[i]:
            #print(data[i + seq], "real: %d predict: %d \n"%(type[i + seq][0], int(pred[i])))
            plt.plot(data[i + seq][0], data[i + seq][1], wrongmode[type[i + seq][0]])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("The Chart of Discriminative Mode %s"%s)
    plt.show()

# Execute statements
if __name__ == '__main__':
    datacreate()
    (size, data, type) = readdata()
    seq = int(size/2)
    (prob, mean, cov, size1, size2, size3) = init_generative(seq, data[0:seq], type)
    pred = predict_generative(seq, data[seq:size], prob, mean, cov)
    test_generative(size, data, type, pred)

    pred = predict_discriminative_one_against_the_rest(seq, data[0 : size1], data[size1 : size1+size2], data[size1+size2 : size1+size2+size3], data[size1+size2+size3 : 2*(size1+size2+size3)])
    test_discriminative(size, data, type, pred, "one against the rest")
    pred = predict_discriminative_pairwise_classification(seq, data[0 : size1], data[size1 : size1+size2], data[size1+size2 : size1+size2+size3], data[size1+size2+size3 : 2*(size1+size2+size3)])
    test_discriminative(size, data, type, pred, "pairwise classification")
