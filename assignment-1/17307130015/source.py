import numpy as np 
import matplotlib.pyplot as plt 
import pdb
import argparse
import random
from models import *


def sample_from_normal_distribution(mean, cov, size): # 只是一个简单的封装
	return np.random.multivariate_normal(mean=mean, cov=cov, size = size)

def render_points(dataset_A,dataset_B,dataset_C):
	plt.plot(dataset_A[:, 0], dataset_A[:, 1], 'r.',label = 'label A', markersize = 3.5)
	plt.plot(dataset_B[:, 0], dataset_B[:, 1], 'b.',label = 'label B', markersize = 3.5)
	plt.plot(dataset_C[:, 0], dataset_C[:, 1], 'g.',label = 'label C', markersize = 3.5)
	plt.legend(loc='lower right', fontsize=8)
	plt.show()

def generate_dataset(mean_A, cov_A, size_A, mean_B, cov_B, size_B, mean_C, cov_C, size_C):

	mean_A = np.array(mean_A)
	cov_A = np.array(cov_A).reshape(2,2)
	dataset_A = sample_from_normal_distribution(mean_A, cov_A, size_A)
	mean_B = np.array(mean_B)
	cov_B = np.array(cov_B).reshape(2,2)
	dataset_B = sample_from_normal_distribution(mean_B, cov_B, size_B)
	mean_C = np.array(mean_C)
	cov_C = np.array(cov_C).reshape(2,2)
	dataset_C = sample_from_normal_distribution(mean_C, cov_C, size_C)

	render_points(dataset_A, dataset_B, dataset_C)

	dataset = np.vstack((dataset_A,dataset_B,dataset_C))
	labels = arg.size_A * [0] + arg.size_B * [1] + arg.size_C * [2] # 0 for A, 1 for B, 2 for C

	return dataset, labels

def save_dataset(dataset, labels, data_path):
	with open(data_path, 'w', encoding = 'utf-8') as f:
		dataset_size = dataset.shape[0]
		for i in range(dataset_size):
			f.write(str(dataset[i][0])+' '+str(dataset[i][1])+' '+str(labels[i])+'\n')

def load_dataset(data_path):
	dataset = []
	labels = []
	dataset_A = []
	dataset_B = []
	dataset_C = []
	datasets = [dataset_A, dataset_B, dataset_C]

	with open(data_path, 'r', encoding = 'utf-8') as f:
		for line in f.readlines():
			line = line[:-1]
			line_ = line.split(' ')
			dataset.append([float(line_[0]), float(line_[1])])
			labels.append(int(line_[-1]))
			datasets[int(line_[-1])].append([float(line_[0]), float(line_[1])])
	
	dataset_A = np.array(dataset_A)
	dataset_B = np.array(dataset_B)
	dataset_C = np.array(dataset_C)
	render_points(dataset_A, dataset_B, dataset_C)
	print('Num of datum: A:{0}, B:{1}, C:{2}'.format(len(dataset_A), len(dataset_B), len(dataset_C)))

	dataset = np.array(dataset)
	labels = np.array(labels)

	return dataset, labels


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mean_A', type=float, nargs='+', default=[0,0])
	parser.add_argument('--cov_A', type=float, nargs='+', default=[10,0,0,10])
	parser.add_argument('--size_A', type=int, default = random.randint(0,10000))
	
	parser.add_argument('--mean_B', type=float, nargs='+', default=[15,12])
	parser.add_argument('--cov_B', type=float, nargs='+', default=[3,0.5,0.5,2])
	parser.add_argument('--size_B', type=int, default = random.randint(0,10000))

	parser.add_argument('--mean_C', type=float, nargs='+', default=[7,-15])
	parser.add_argument('--cov_C', type=float, nargs='+', default=[3,2,2,3])
	parser.add_argument('--size_C', type=int, default = random.randint(0,10000))

	parser.add_argument('--batch_size', type=int, default = 12)
	parser.add_argument('--l_r', type=float, default = 0.001)
	parser.add_argument('--epoch', type=int, default = 10)
	parser.add_argument('--lambda_regularization', type=float, default = 1e-8)

	parser.add_argument('--seed', type=int, default = 0)
	parser.add_argument('--load_data', default= False,action='store_true')
	parser.add_argument('--use_generative', default= False,action='store_true')
	parser.add_argument('--dataset', type=str ,default= 'dataset.data')
	parser.add_argument('--split_rate', type=float ,default= 0.95)

	parser.add_argument('--layers_dims', type=int, nargs='+', default = [2,20,7,5,3])

	arg = parser.parse_args()
	np.random.seed(arg.seed)
	random.seed(arg.seed)

	if arg.load_data == False:
		dataset, labels = generate_dataset(arg.mean_A, arg.cov_A, arg.size_A, arg.mean_B, arg.cov_B, arg.size_B, arg.mean_C, arg.cov_C, arg.size_C)
		random_map = random.sample(range(len(dataset)),len(dataset))
		dataset = dataset[random_map] # 对生成的数据进行打乱
		labels = np.array(labels)[random_map]
		save_dataset(dataset, labels, arg.dataset)
	else:
		dataset, labels = load_dataset(arg.dataset)

	split_rate = arg.split_rate
	split_index = int(len(dataset) * split_rate)
	train_dataset = dataset[ :split_index]
	train_labels = labels[ :split_index]
	test_dataset = dataset[split_index: ]
	test_labels = labels[split_index: ]

	batch_size = arg.batch_size
	l_r = arg.l_r
	epoch = arg.epoch

	print('Batch_size:{0}, l_r:{1}'.format(batch_size, l_r))

	if not arg.use_generative:
		model = FNN(batch_size=arg.batch_size, 
					l_r = arg.l_r,
					lambda_regularization = arg.lambda_regularization,
					layers_dims = arg.layers_dims)

		for epc in range(epoch):
			random_map = random.sample(range(len(train_dataset)),len(train_dataset))
			train_batch_list = [random_map[i:i+batch_size] for i in range(0, len(train_dataset)+1-batch_size, batch_size)]

			total_loss = 0
			total_accuracy = 0
			for i,batch in enumerate(train_batch_list):
				inputs = train_dataset[batch].T
				labels = train_labels[batch]
				outputs = model.forward(inputs)
				preds , cross_entropy_loss, accuracy, loss = model.get_loss_accuracy(outputs, labels)
				model.backward(outputs, labels)
				total_loss += loss * len(batch)
				total_accuracy += accuracy * len(batch)

			print('Train : Epoch {0} Accuracy:{1} Loss:{2}'.format(
				epc, total_accuracy / len(train_dataset), total_loss / len(train_dataset)))

			order_map = [i for i in range(len(test_dataset))]
			test_batch_list = [order_map[i:i+batch_size] for i in range(0, len(test_dataset)+1-batch_size, batch_size)]

			total_loss = 0
			total_accuracy = 0

			for batch in test_batch_list:
				inputs = test_dataset[batch].T
				labels = test_labels[batch]
				outputs = model.forward(inputs)
				preds , cross_entropy_loss, accuracy, loss = model.get_loss_accuracy(outputs, labels)
				total_loss += loss * len(batch)
				total_accuracy += accuracy * len(batch)

			print('Test : Epoch {0} Accuracy:{1} Loss:{2}'.format(
				epc, total_accuracy / len(test_dataset), total_loss / len(test_dataset)))

	else:
		model = NaiveBayersClassifier( num_class = 3, input_dims = 2)
		model.maximum_likelihood_estimation(train_dataset, train_labels)
		preds = model.forward(test_dataset.T)
		accuracy = model.get_accuracy(preds, test_labels)
		print('Test Accuracy: ',accuracy)

