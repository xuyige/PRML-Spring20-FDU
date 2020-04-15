import random
from pprint import pformat
import numpy as np
import pdb
from copy import deepcopy

def softmax(y):
	return np.exp(y) / (np.sum(np.exp(y)))
def sigmoid(y):

	return np.exp(y) / (1 + np.exp(y))

class DiscriminativeModel:
	def __init__(self , d , ntype):

		self.d = d
		self.W = np.ndarray(shape = (ntype , d+1))

		self.reset_param()

	def reset_param(self):
		n , m = self.W.shape
		for i in range(n):
			for j in range(m):
				self.W[i , j] = np.random.normal(0 , 1e-5)

	def forward(self , x , label = None):

		x = np.insert(x , self.d , values = 1 , axis = 0)

		y_hat = np.matmul(self.W , x)

		y = softmax(y_hat)

		self.y = y
		self.x = x

		return y + 1e-6

	def backward(self , label):
		dLdy_hat = self.y.copy()
		dLdy_hat[label] -= 1
		x = self.x
		return np.matmul(dLdy_hat , x.T) 

class GenerativeModel:
	def __init__(self , d , ntype):

		self.d = d
		self.ntype = ntype
		self.W = np.ndarray(shape = (ntype , d+1))

		self.reset_param()

	def reset_param(self):
		n , m = self.W.shape
		for i in range(n):
			for j in range(m):
				self.W[i , j] = np.random.normal(0 , 1e-5)

	def forward(self , x , label = None):

		x = np.insert(x , self.d , values = 1 , axis = 0)

		y_hat = np.matmul(self.W , x)

		y = sigmoid(y_hat)

		self.y = y.copy()
		self.x = x.copy()

		return y

	def backward(self , label):
		dLdy_hat = self.y.copy()
		dLdy_hat[label] -= 1
		x = self.x.copy()
		#pdb.set_trace()
		return np.matmul(dLdy_hat , x.T) 
