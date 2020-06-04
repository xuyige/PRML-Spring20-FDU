import numpy as np 
import matplotlib.pyplot as plt 
import pdb
import argparse
import random
import handout.models as models

def sample_from_normal_distribution(mean, cov, size): # 只是一个简单的封装
	return np.random.multivariate_normal(mean=mean, cov=cov, size = size)

def render_points(dataset_A,dataset_B,dataset_C):
	plt.plot(dataset_A[:, 0], dataset_A[:, 1], 'r.',label = 'label A', markersize = 3.5)
	plt.plot(dataset_B[:, 0], dataset_B[:, 1], 'b.',label = 'label B', markersize = 3.5)
	plt.plot(dataset_C[:, 0], dataset_C[:, 1], 'g.',label = 'label C', markersize = 3.5)
	plt.legend(loc='lower right', fontsize=8)
	plt.show()

def render_points_all(dataset):
	plt.plot(dataset[:, 0], dataset[:, 1], '.',color = 'violet',label = 'dataset', markersize = 3.5)
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
	render_points_all(np.vstack([dataset_A, dataset_B, dataset_C]))

	dataset = np.vstack((dataset_A,dataset_B,dataset_C))
	labels = size_A * [0] + size_B * [1] + size_C * [2] # 0 for A, 1 for B, 2 for C

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
	render_points_all(np.vstack([dataset_A, dataset_B, dataset_C]))
	print('Num of datum: A:{0}, B:{1}, C:{2}'.format(len(dataset_A), len(dataset_B), len(dataset_C)))

	dataset = np.array(dataset)
	labels = np.array(labels)

	return dataset, labels

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mean_A', type=float, nargs='+', default=[0,0])
	parser.add_argument('--cov_A', type=float, nargs='+', default=[10,0,0,10])
	parser.add_argument('--prob_A', type=float, default = 0.28)
	
	parser.add_argument('--mean_B', type=float, nargs='+', default=[15,12])
	parser.add_argument('--cov_B', type=float, nargs='+', default=[3,0.5,0.5,2])
	parser.add_argument('--prob_B', type=float, default = 0.48)

	parser.add_argument('--mean_C', type=float, nargs='+', default=[7,-15])
	parser.add_argument('--cov_C', type=float, nargs='+', default=[3,2,2,3])
	parser.add_argument('--prob_C', type=float, default = 0.24)

	parser.add_argument('--dataset_size', type=int, default = 100000);

	parser.add_argument('--seed', type=int, default = 0)
	parser.add_argument('--load_data', default= False,action='store_true')
	parser.add_argument('--dataset', type=str ,default= 'dataset.data')
	parser.add_argument('--print_every', default= False, action='store_true')

	arg = parser.parse_args()
	np.random.seed(arg.seed)
	random.seed(arg.seed)

	if arg.load_data == False:
		uniform_dataset = np.random.uniform(0, 1, arg.dataset_size)
		size_A = np.sum(uniform_dataset < arg.prob_A)
		size_C = np.sum(uniform_dataset > 1-arg.prob_C)
		size_B = arg.dataset_size - size_A - size_C
		print("Size A: {0}, Size B: {1}, Size C: {2}".format(size_A, size_B, size_C))
		#size_A, size_B, size_C = (int)(arg.dataset_size*arg.prob_A), (int)(arg.dataset_size*arg.prob_B), (int)(arg.dataset_size*arg.prob_C)
		dataset, labels = generate_dataset(arg.mean_A, arg.cov_A, size_A, arg.mean_B, arg.cov_B, size_B, arg.mean_C, arg.cov_C, size_C)
		random_map = random.sample(range(len(dataset)),len(dataset))
		dataset = dataset[random_map] # 对生成的数据进行打乱
		labels = np.array(labels)[random_map]
		save_dataset(dataset, labels, arg.dataset)
	else:
		dataset, labels = load_dataset(arg.dataset)


	model = models.GMM(num_classes = 3, input_dims = 2, inputs = dataset)
	for i in range(100):
		print("Epoch : ",i)
		model.E_step()
		model.M_step()
		dif = model.converge()
		if arg.print_every or (dif < arg.dataset_size / 100000) :
			model.render_partition()
			if (dif < arg.dataset_size / 100000):
				break

