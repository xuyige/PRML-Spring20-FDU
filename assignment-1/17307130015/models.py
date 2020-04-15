import numpy as np 
import pdb
import math

def sigmoid(z):

	return 1 / (1+ np.exp(-z))

def relu(z):
	return np.maximum(0, z)

def softmax(z):
	z = np.exp(z)
	z_sum_col = np.sum(z, axis = 0, keepdims = True)
	z = z / z_sum_col
	return z

class FNN(object):
	def __init__(self, batch_size, l_r, lambda_regularization = 1e-8,layers_dims = [2,20,7,5,3]):
		# layers_dims e.g. 2,20,7,5,3 则20是输入的维数，3是输出的维数
		# 输入的shape是 2 (高斯分布的维数) * batch_size，每一个样本是一个列向量
		self.batch_size = batch_size
		self.l_r = l_r
		self.layers_dims = layers_dims
		self.num_layers = len(layers_dims) - 1
		self.Weight = [i for i in range(self.num_layers+1)] #但用不到Weight[0]
		self.bias = [i for i in range(self.num_layers+1)] #但用不到Bias[0]
		self.lambda_regularization = lambda_regularization
		self.varis = {} # 中间变量
		self.grads = {}

		for i in range(self.num_layers):
			self.Weight[i+1] = np.random.randn(layers_dims[i+1], layers_dims[i]) * np.sqrt(2 / (layers_dims[i+1] + layers_dims[i]))
			self.bias[i+1] = np.zeros((layers_dims[i+1], 1))

	def forward(self, inputs):
		# 输入的shape是 2 (高斯分布的维数) * batch_size，每一个样本是一个列向量


		Z = [i for i in range(self.num_layers+1)] # 但不用到z[0] 大写的Z表示矩阵（batch），小写的z表示向量（单样本）
		A = [i for i in range(self.num_layers+1)] # a[0]即inputs

		A[0] = inputs
		
		for i in range(self.num_layers):
			Z[i+1] = np.matmul( self.Weight[i+1], A[i]) + self.bias[i+1]
			if i+1 == self.num_layers:
				# 最后一层 使用softmax
				A[i+1] = softmax(Z[i+1])
			else:
				A[i+1] = relu(Z[i+1])

		self.varis['Z'] = Z
		self.varis['A'] = A

		return A[self.num_layers]

	def get_loss_accuracy(self, outputs, labels):
		# outputs : num_class(3) * batch_size, softmax的输出，每一个列向量其内部和为1
		# labels ： batch_size，每一位为0/1/2
		
		# 计算交叉熵：对output取log，与labels对应列进行点积，取负，再对batch_size取平均

		labels_onehot = np.eye(self.layers_dims[-1])[labels].T
		cross_entropy_loss = -np.mean(np.sum(np.multiply(np.log(outputs), labels_onehot), axis=0))
		preds = outputs.argmax(axis= 0)
		accuracy = np.mean(np.equal(preds, labels))

		regularization_loss = 0
		for i in range(1,self.num_layers+1):
			regularization_loss = regularization_loss + np.sum(np.square(self.Weight[i]))
		regularization_loss = regularization_loss* (self.lambda_regularization / 2) /self.batch_size

		loss = regularization_loss + cross_entropy_loss

		# 交叉熵是1*1的标量，outputs是num_class * batch_size，取导数为num_class * batch_size的形式
		# 则 dCrossEntropy = np.sum(np.multiply(dCrossEntropy_dOutputs, d_outputs))
		

		return preds, cross_entropy_loss, accuracy, loss

	def backward(self, outputs, labels):

		labels_onehot = np.eye(self.layers_dims[-1])[labels].T
		dCrossEntropy_dOutputs = -1/self.batch_size * np.divide(labels_onehot, outputs)

		Z = self.varis['Z']
		A = self.varis['A']

		
		dLoss_dZ = [0] + [ np.zeros(Z[i].shape) for i in range(1, self.num_layers+1)] # 因为用不到Z[0]，所以这边的第0项也用不到；dloss_dZ[l][b] 表示loss对Z[l][:,b]的导数
		dLoss_dA = [0] + [ np.zeros(A[i].shape) for i in range(1, self.num_layers+1)] # 用不到第0项，最后一项即loss对outputs求导
		dLoss_dW = [ i for i in range(self.num_layers+1)]
		dLoss_db = [ i for i in range(self.num_layers+1)]

		_ = [j for j in range(self.batch_size)]
		dA_dZ = [ _.copy() for i in range(self.num_layers+1)] # 对每一层i，对每一个样本（第j列），aj和zj都是3维向量，因而daj_dzj是3x3的矩阵；第0层除外
		dZ_dA = [ i for i in range(self.num_layers+1)]
		dZ_dW = [ i for i in range(self.num_layers+1)]

		dLoss_dA[-1] = dCrossEntropy_dOutputs

		for i in reversed(range(1, self.num_layers+1)): # i表示层数，第1层到第L层
			if i == self.num_layers:
				for j in range(self.batch_size): #j表示batch中的样本序号
					# a是softmax的结果，z是softmax前的logits，保持a和z为列向量
					a = A[i][:, j].reshape(1,-1).T
					z = Z[i][:, j].reshape(1,-1).T
					
					dA_dZ[i][j] = np.diag(a.reshape(self.layers_dims[-1])) - np.matmul(a, a.T) 
					dLoss_dZ[i][:, j] = np.matmul(dA_dZ[i][j], dLoss_dA[i][:, j]) 
			else:
				dA_dZ[i] = (Z[i] > 0) + 0
				dLoss_dZ[i] = np.multiply(dLoss_dA[i], dA_dZ[i])

			dZ_dA[i] = self.Weight[i]# 即dZi 对dAi-1的导数
			dZ_dW[i] = A[i-1]

			dLoss_dA[i-1] = np.matmul(dZ_dA[i].T, dLoss_dZ[i])
			dLoss_dW[i] = np.matmul(dLoss_dZ[i], dZ_dW[i].T) + \
				(self.lambda_regularization / self.batch_size) * self.Weight[i]# 第一项是交叉熵对W的导数，第二项是正则项的导数
			dLoss_db[i] = np.sum(dLoss_dZ[i], axis =1, keepdims = True)

			
			self.Weight[i] = self.Weight[i] - self.l_r * dLoss_dW[i]
			self.bias[i] = self.bias[i] - self.l_r * dLoss_db[i]

class NaiveBayersClassifier(object):
	def __init__(self, num_class, input_dims):
		
		self.num_class = num_class # 类的个数，即分布的个数
		self.input_dims = input_dims
		self.mean = []
		self.cov = []
		self.cov_inv = []
		self.prior = [0 for i in range(num_class)]
		for i in range(num_class):
			# 因为输入inputs是列向量，故将mean也设为列向量
			mean = np.zeros((input_dims, 1))
			cov = np.zeros((input_dims, input_dims))
			cov_inv = np.zeros((input_dims, input_dims))
			self.mean.append(mean)
			self.cov.append(cov)
			self.cov_inv.append(cov_inv)

	def maximum_likelihood_estimation(self, dataset, labels):
		size = [ 0 for i in range(self.num_class)] # size[i]表示第i个分布的点的数量
		for i,dot in enumerate(dataset):
			label = labels[i]
			size[label] += 1
			self.mean[label] = self.mean[label] + dot.reshape((-1,1))

		for i in range(self.num_class):
			self.mean[i] = self.mean[i] / size[i]

		for i,dot in enumerate(dataset):
			label = labels[i]
			dot = dot.reshape((-1,1))
			self.cov[label] = self.cov[label] + np.dot((dot - self.mean[label]), (dot - self.mean[label]).T)

		for i in range(self.num_class):
			self.cov[i] = self.cov[i] / size[i]
			self.cov_inv[i] = np.linalg.inv(self.cov[i])
			self.prior[i] = size[i] / len(dataset)

	def forward(self, inputs):

		prior = np.array([i for i in self.prior]).reshape(self.num_class,1)
		likelihood = np.zeros((self.num_class, len(inputs.T)))

		for i, dot in enumerate(inputs.T):
			for c in range(self.num_class):
				likelihood[c][i] = ((2*math.pi)**(-self.input_dims/2)) * \
									(np.linalg.det(self.cov[c])**(-1/2)) * \
									math.exp( (-1/2) *  np.matmul( 
										np.matmul( (dot - self.mean[c].reshape(-1)).reshape(1,-1),  self.cov_inv[c]), 
										(dot - self.mean[c].reshape(-1)).reshape(-1,1)) )
		posterior = prior * likelihood 	#将进行广播，变成likelihood的每一列与prior进行对应元素相乘
		pred = posterior.argmax(axis = 0)

		return pred

	def get_accuracy(self, preds, labels):
		accuracy = np.mean(np.equal(preds, labels))
		return accuracy


		



			




				


				


