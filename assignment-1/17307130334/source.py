import copy
import csv
import math
import random
import sys
import time

from matplotlib import pyplot as plt
from numpy.random import multivariate_normal as mvn

# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def gendata():
    # Generate 3 gaussian distribution
    sz=100
    mean_1 = [4, 3]
    mean_2 = [0, 0]
    mean_3 = [-3, 4]
    cov_1 = [[1, 0], [0, 1]]
    cov_2 = [[1, 0], [0, 1]]
    cov_3 = [[1, 0], [0, 1]]
    dist_1 = mvn(mean_1, cov_1, sz).T
    dist_2 = mvn(mean_2, cov_2, sz).T
    dist_3 = mvn(mean_3, cov_3, sz).T
    # Write data out
    data = []
    for i in range(0, sz):
        data.append([dist_1[0][i], dist_1[1][i], 1])
        data.append([dist_2[0][i], dist_2[1][i], 2])
        data.append([dist_3[0][i], dist_3[1][i], 3])
    random.shuffle(data)
    with open('data.data', 'w')as f:
        writer = csv.writer(f)
        writer.writerows(data)
    # Plot
    plt.plot(dist_1[0], dist_1[1], 'x', color='r')
    plt.plot(dist_2[0], dist_2[1], 'x', color='b')
    plt.plot(dist_3[0], dist_3[1], 'x', color='y')
    plt.axis('equal')
    plt.show()

def getdata():
    with open('data.data')as f:
        reader=csv.reader(f, delimiter=',')
        data = list([float(d[0]),float(d[1]),int(d[2])] for d in reader)
    return data

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds,*args):

    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    times=list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list(fold)
        actual = [row[-1] for row in fold]
        start=time.time()
        predicted = algorithm(train_set, test_set,*args)
        end=time.time()
        # myplot(test_set, predicted)
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
        times.append(end-start)
    return scores,times

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        # Calculate the mean, stdev and count for each column in a dataset
        summary=[(mean(column), stdev(column), len(column)) for column in zip(*rows)]
        del (summary[-1])
        summaries[class_value] = summary
    return summaries

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        # P(class)
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

# Naive Bayes Algorithm
def naive_bayes(train, test):
    summaries = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summaries, row)
        predictions.append(output)
    return (predictions)

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset):
    # Find the min and max values for each column
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])

    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Make a prediction with coefficients
def dis_predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return sigmoid(yhat)

def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = dis_predict(row, coef)
            error = row[-1] - yhat
            sum_error += error ** 2
            coef[0] = coef[0] + l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] + l_rate * error * row[i]
    return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    coefs=list()
    for i in range(0,3):
        train_copy=copy.deepcopy(train)
        for row in train_copy:
            if(row[-1]==i+1):
                row[-1]=1
            else:
                row[-1]=0
        coef = coefficients_sgd(train_copy, l_rate, n_epoch)
        coefs.append(coef)
    for row in test:
        maxp=-1
        label=0
        for i in range(0,3):
            yhat = dis_predict(row, coefs[i])
            if(yhat>maxp):
                maxp=yhat
                label=i+1
        predictions.append(label)
    return (predictions)

def main(argv):
    if(argv[1]=='gendata'):
        gendata()
    elif (argv[1]=='generative_model'):
        n_folds = 5
        dataset = getdata()
        scores,times = evaluate_algorithm(dataset, naive_bayes, n_folds)
        print('Scores: %s' % scores)
        # print('Times:%s '%times)
        print('Mean Accuracy: %.2f%%' % (sum(scores) / float(len(scores))))
        print('Mean Running Time:%.3f'%(sum(times) / float(len(times))))
    elif (argv[1] == 'discriminative_model'):
        dataset = getdata()
        # normalize
        normalize_dataset(dataset)
        # evaluate algorithm
        n_folds = 5
        l_rate = 0.1
        n_epoch = 100
        scores,times = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
        print('Scores: %s' % scores)
        # print('Times:%.3f ' % times)
        print('Mean Accuracy: %.2f%%' % (sum(scores) / float(len(scores))))
        print('Mean Running Time:%.3f' % (sum(times) / float(len(times))))
    else:
        print("Invalid command.")

if __name__ == '__main__':
    main(sys.argv)