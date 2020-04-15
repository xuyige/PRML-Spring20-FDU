import numpy as np
from scipy.stats import multivariate_normal
class DiscriminativeModel:
	def __init__(self):
		self.W = None

	def loss(self, X, y, reg):
		'''
		返回交叉熵损失函数的值loss和W的梯度dW
		'''
		loss = 0
		W = self.W
		X = np.hstack((X, np.ones((X.shape[0], 1))))
		dW = np.zeros_like(W)

		Scores = X.dot(W)
		Scores = Scores - np.max(Scores, axis = 1, keepdims = True)
		ExpScores = np.exp(Scores)

		loss = -np.sum(Scores[np.arange(X.shape[0]), y]) + np.sum(np.log(np.sum(ExpScores, axis = 1)))
		loss /= X.shape[0]
		loss += 0.5 * reg * np.sum(W * W)

		tmp = np.zeros((X.shape[0], W.shape[1]))
		tmp[np.arange(X.shape[0]), y] = -1
		Frac = ExpScores / np.sum(ExpScores, axis = 1, keepdims = True)
		dW += X.T.dot(Frac + tmp)
		dW /= X.shape[0]
		dW += reg * W

		'''
		y_pred = np.argmax(X.dot(self.W), axis = 1)
		ret = np.mean(y == y_pred)

		print(ret)
		'''

		return loss, dW

	def predict(self, X):
		'''
		返回预测结果y_pred
		'''
		X = np.hstack((X, np.ones((X.shape[0], 1))))
		return np.argmax(X.dot(self.W), axis = 1)

	def train(self, X, y, lr = 1e-3, reg = 1e-5, nIters = 1000):
		'''
		训练该模型，lr表示学习速率，reg表示正则化强度，nIters表示sgd的训练组数
		'''
		nTrain, dim = X.shape
		nClass = np.max(y) + 1
		if self.W == None:
			self.W = 0.001 * np.random.randn(dim + 1, nClass)
		# print(self.W)
		loss_history = []
		for it in range(nIters):
			Indexs = np.random.choice(nTrain, 800)
			X_batch = X[Indexs]
			y_batch = y[Indexs]

			loss, grad = self.loss(X_batch, y_batch, reg)
			self.W += - lr * grad
			# print(loss)

		# print(self.W)

class GenerativeModel:
	def __init__(self):
		self.prior = None
		self.mean = None
		self.cov = None

	def train(self, X, y):
		'''
		通过统计得到prior,mean和cov
		'''
		nTrain, dim = X.shape
		nClass = np.max(y) + 1

		self.mean = np.zeros((nClass, dim))
		self.cov = np.zeros((nClass, dim, dim))

		self.prior = np.bincount(y)
		for i in range(nClass):
			self.mean[i] = np.sum(X[y == i], axis = 0) / self.prior[i]

		for i in range(nTrain):
			self.cov[y[i]] += np.dot((X[i] - self.mean[y[i]]).reshape(dim, 1), (X[i] - self.mean[y[i]]).reshape(1, dim))
		
		for i in range(nClass):
			self.cov[i] /= self.prior[i]

		self.prior = self.prior.astype('float64') / nTrain

	def predict(self, X):
		'''
		分别尝试不同的高斯分布，取概率最大的为结果。
		返回预测值y_pred
		'''
		nTrain, dim = X.shape
		nClass = self.cov.shape[0]
		prob = np.zeros((nClass, nTrain))

		for i in range(nClass):
			prob[i] = multivariate_normal.pdf(X, self.mean[i], self.cov[i]) * self.prior[i]

		y_pred = np.argmax(prob, axis = 0)
		return y_pred

