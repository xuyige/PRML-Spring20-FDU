import numpy as np
from handout import datatools
from scipy.stats import multivariate_normal

class GMM():
	def __init__(self, n_class = 3, dim = 2):
		self.n_class = n_class
		self.dim = dim
		self.mean = np.zeros(n_class * dim).reshape((n_class, dim))
		self.cov = np.zeros(n_class * dim * dim).reshape((n_class, dim, dim))
		for i in range(n_class):
			self.cov[i] = np.eye(dim)
		self.prior = np.ones(n_class) / n_class

	def k_means(self, data, iters = 10):
		n = data.shape[0]
		
		for cnt in range(iters):
			mean = np.zeros(self.n_class * self.dim).reshape((self.n_class, self.dim))
			num = np.zeros(self.n_class)
			for i in range(n):
				diff = np.zeros(self.n_class)
				for j in range(self.n_class):
					diff[j] = np.sum((data[i] - self.mean[j]) ** 2)
				tmp = np.argmin(diff)
				mean[tmp] += data[i]
				num[tmp] += 1
			mean = mean / num.reshape((self.n_class, 1))

			if np.sum((mean - self.mean) ** 2) < 0.1:
				break
			self.mean = mean.copy()
		# print(self.mean)


	def train(self, data, iters = 20, use_kmeans = True):
		n = data.shape[0]
		self.mean = data[np.random.choice(n, self.n_class, False)]

		if use_kmeans:
			self.k_means(data) # use k-means to initiate

		for Iter in range(iters):
			gamma = np.zeros((n, self.n_class))
			for i in range(self.n_class):
				gamma[:, i] = multivariate_normal.pdf(data, self.mean[i], self.cov[i]) * self.prior[i]
			gamma = gamma / np.sum(gamma, axis = 1, keepdims = True)

			N = np.sum(gamma, axis = 0)
			self.prior = N / n
			for i in range(self.n_class):
				self.mean[i] = 1/N[i] * np.sum(gamma[:, i].reshape(-1, 1) * data, axis = 0)
				self.cov[i] = np.diag(1/N[i] * np.sum(gamma[:, i].reshape(-1, 1) * (data - self.mean[i].reshape(1, self.dim)) ** 2, axis = 0))

	def predict(self, X):
		n = X.shape[0]
		n_class = self.n_class
		prob = np.zeros((n_class, n))

		for i in range(n_class):
			prob[i] = multivariate_normal.pdf(X, self.mean[i], self.cov[i]) * self.prior[i]

		y_pred = np.argmax(prob, axis = 0)
		return y_pred

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
	data1, data2, data3 = datatools.workdata(True, mean1, cov1, num1, mean2, cov2, num2, mean3, cov3, num3, need_save = True)

	#data1 = np.hstack((data1, 0 * np.ones(data1.shape[0]).reshape(data1.shape[0], 1)))
	#data2 = np.hstack((data2, 1 * np.ones(data2.shape[0]).reshape(data2.shape[0], 1)))
	#data3 = np.hstack((data3, 2 * np.ones(data3.shape[0]).reshape(data3.shape[0], 1)))

	data = np.vstack((data1, data2, data3))
	#np.random.shuffle(data)
	#X = data[np.arange(data.shape[0]), :2]
	#Y = data[np.arange(data.shape[0]), -1].astype('int')

	# print(X.shape)
	gmm = GMM(3, data.shape[1])
	gmm.train(data, use_kmeans = True)

	for i in range(gmm.n_class):
		print('Guass %d' % i)
		print('mean:')
		print(gmm.mean[i])
		print('cov:')
		print(gmm.cov[i])
		print('prior')
		print(gmm.prior[i])


	Y = gmm.predict(data)

	res = [[], [], []]
	for i in range(data.shape[0]):
		res[Y[i]].append(data[i])
	datatools.draw(np.array(res[0]), np.array(res[1]), np.array(res[2]))
