import numpy as np

def gaussian(mean, cov, num):
	return np.random.multivariate_normal(mean, cov, num)

def makedata(num1, num2, num3):
	mean1 = [-5, 5]
	mean2 = [0, 5]
	mean3 = [5, 0]

	cov1 = [[1, 0], [0, 1]]
	cov2 = [[1, 0], [0, 1]]
	cov3 = [[1, 0], [0, 1]]

	data1 = np.append(gaussian(mean1, cov1, num1), np.ones(num1, int).reshape(num1, 1), axis = 1)
	data2 = np.append(gaussian(mean2, cov2, num2), np.ones(num2, int).reshape(num2, 1) * 2, axis = 1)
	data3 = np.append(gaussian(mean3, cov3, num3), np.ones(num3, int).reshape(num3, 1) * 3, axis = 1)

	data = np.vstack((data1, data2, data3))

	np.random.shuffle(data)

	return data

