import numpy as np 
import math
import pdb
import matplotlib.pyplot as plt 

def calc_prob(inputs, mean, cov): # 计算正态分布的概率密度
	input_dims = mean.shape[-1] 
	dataset_size = inputs.shape[0]
	prob = np.zeros((dataset_size, 1))
	inputs = np.expand_dims(inputs, axis = 1)
	prob = (np.exp((-1/2)*(np.matmul(np.matmul(inputs-mean, np.linalg.inv(cov)), (inputs-mean).reshape(dataset_size, input_dims, 1))))).reshape(dataset_size, 1) / \
			(math.pow(2*math.pi, input_dims/2) * np.sqrt(abs(np.linalg.det(cov))))
	
	'''
	for i in range(dataset_size):
		prob[i] = (math.exp((-1/2)*(np.matmul(np.matmul(inputs[i]-mean, np.linalg.inv(cov)), (inputs[i]-mean).T)))) / \
			(math.pow(2*math.pi, input_dims/2) * np.sqrt(abs(np.linalg.det(cov))))
	'''

	#pdb.set_trace()
	return prob

def render_points(dataset_A,dataset_B,dataset_C):

	if len(dataset_A) > 0:
		plt.plot(dataset_A[:, 0], dataset_A[:, 1], 'r.',label = 'label A', markersize = 3.5)
	if len(dataset_B) > 0:
		plt.plot(dataset_B[:, 0], dataset_B[:, 1], 'b.',label = 'label B', markersize = 3.5)
	if len(dataset_C) > 0:
		plt.plot(dataset_C[:, 0], dataset_C[:, 1], 'g.',label = 'label C', markersize = 3.5)
	plt.legend(loc='lower right', fontsize=8)
	plt.show()

class GMM(object):
	def __init__(self, num_classes, input_dims, inputs):
		self.num_classes = num_classes
		self.input_dims = input_dims
		self.mean = np.random.randn(num_classes, input_dims)
		self.cov = np.zeros((num_classes, input_dims, input_dims))
		self.prior = np.ones(num_classes) / num_classes # 多项分布的概率 取哪个正态分布
		self.inputs = inputs
		self.dataset_size = inputs.shape[0]
		self.nk = np.array([ self.dataset_size / num_classes for i in range(num_classes)])
		for i in range(num_classes):
			self.cov[i] = np.eye(input_dims)

	def E_step(self):
		# 输入的格式应该为 [dataset_size, input_dims]
		inputs = self.inputs
		dataset_size = inputs.shape[0]
		normal_density = np.zeros((inputs.shape[0], self.num_classes))
		for i in range(self.num_classes):
			normal_density[:, i] = calc_prob(inputs, self.mean[i], self.cov[i]).reshape(dataset_size)
		posterior = self.prior * normal_density
		self.posterior = posterior / np.sum(posterior, axis=1).reshape(dataset_size, 1)

	def M_step(self):
		#pdb.set_trace()
		self.pre_nk = self.nk
		self.nk = np.sum(self.posterior, axis=0) # 每个类的样本期望值
		self.prior = self.nk / self.dataset_size # 取到每个类的概率期望
		self.mean = (np.matmul(self.inputs.T, self.posterior) / self.nk).T
		for i in range(self.num_classes):
			mean = self.mean[i]
			dif = np.expand_dims(self.inputs - mean, axis = 2)
			posterior = self.posterior[:, i].reshape(self.dataset_size, 1, 1)
			self.cov[i] = np.sum( posterior * np.matmul(dif, dif.reshape(self.dataset_size, 1, self.input_dims)), axis=0) / self.nk[i]

	def render_partition(self):
		'''
		print("mean \n", self.mean)
		print("cov \n", self.cov)
		print("prior \n", self.prior)
		'''
		pred = np.argmax(self.posterior, axis = 1)
		dataset_A = []
		dataset_B = []
		dataset_C = []
		datasets = [dataset_A, dataset_B, dataset_C]
		for i in range(self.dataset_size):
			datasets[pred[i]].append(self.inputs[i])
		dataset_A = np.vstack(dataset_A) if (len(dataset_A) > 0) else dataset_A
		dataset_B = np.vstack(dataset_B) if (len(dataset_B) > 0) else dataset_B
		dataset_C = np.vstack(dataset_C) if (len(dataset_C) > 0) else dataset_C

		render_points(dataset_A, dataset_B, dataset_C)

	def converge(self):
		dif = np.sum(abs(self.nk - self.pre_nk))
		print("L :",dif)
		return dif

		
