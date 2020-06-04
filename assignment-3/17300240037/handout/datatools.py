import numpy as np
import matplotlib.pyplot as plt


def draw(data1, data2, data3):
	plt.plot(data1.T[0], data1.T[1], 'r.')
	plt.plot(data2.T[0], data2.T[1], 'g.')
	plt.plot(data3.T[0], data3.T[1], 'b.')

	mean = np.mean(data1, axis = 0)
	plt.plot(mean[0], mean[1], 'x')
	mean = np.mean(data2, axis = 0)
	plt.plot(mean[0], mean[1], 'x')
	mean = np.mean(data3, axis = 0)
	plt.plot(mean[0], mean[1], 'x')
	plt.show()

def gen(mean1, cov1, num1, mean2, cov2, num2, mean3, cov3, num3):
	data1 = np.random.multivariate_normal(mean1, cov1, num1)
	data2 = np.random.multivariate_normal(mean2, cov2, num2)
	data3 = np.random.multivariate_normal(mean3, cov3, num3)

	return data1, data2, data3

def save(data1, data2, data3, filename = 'GMM.data'):
	out = ''
	out += str(data1.shape[0]) + '\n'
	for i in range(data1.shape[0]):
		out += str(data1[i][0]) + ' ' + str(data1[i][1]) + '\n'
	out += str(data2.shape[0]) + '\n'
	for i in range(data2.shape[0]):
		out += str(data2[i][0]) + ' ' + str(data2[i][1]) + '\n'
	out += str(data3.shape[0]) + '\n'
	for i in range(data3.shape[0]):
		out += str(data3[i][0]) + ' ' + str(data3[i][1]) + '\n'
	with open(filename, 'w') as f:
		f.write(out)

def load(filename = 'GMM.data'):
	d1 = []
	d2 = []
	d3 = []
	with open(filename, 'r') as f:
		n1 = int(f.readline())
		for i in range(n1):
			d1.append([float(x) for x in f.readline().split(' ')])

		n2 = int(f.readline())
		for i in range(n2):
			d2.append([float(x) for x in f.readline().split(' ')])

		n3 = int(f.readline())
		for i in range(n3):
			d3.append([float(x) for x in f.readline().split(' ')])
	d1 = np.array(d1)
	d2 = np.array(d2)
	d3 = np.array(d3)
	return d1, d2, d3


def workdata(need_gen, mean1 = 0, cov1 = 0, num1 = 0, mean2 = 0, cov2 = 0, num2 = 0, mean3 = 0, cov3 = 0, num3 = 0, filename = 'GMM.data', need_save = False):
	if need_gen == True:
		data1, data2, data3 = gen(mean1, cov1, num1, mean2, 
								  cov2, num2, mean3, cov3, num3)
		if need_save == True:
			save(data1, data2, data3, filename)
		'''
		# to check the correctness of save and load
		d1, d2, d3 = load(filename)
		print(np.sum(np.abs(d1 - data1)) + np.sum(np.abs(d2 - data2)) + np.sum(np.abs(d3 - data3)))
		'''
		draw(data1, data2, data3)
		return data1, data2, data3
	else:
		data1, data2, data3 = load(filename)
		draw(data1, data2, data3)
		return data1, data2, data3

if __name__ == '__main__':
	mean1 = [0, 0]
	mean2 = [5, 5]
	mean3 = [5, 0]
	cov1 = [[1, 0],
			[0, 1]]
	cov2 = [[1, 0],
			[0, 1]]
	cov3 = [[1, 0],
			[0, 1]]
	num1 = 150
	num2 = 150
	num3 = 150
	workdata(True, mean1, cov1, num1, mean2, cov2, num2, mean3, cov3, num3)