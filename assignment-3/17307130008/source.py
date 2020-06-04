from handout import *
import numpy as np
import matplotlib as plot

"""
运行方法：python source.py
"""

mean = np.array([[-5, -5], [-1, 5], [5, 0]])
cov = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])
sample = normal_distribution_generate(mean, cov, 3, [100, 100, 100])
save_dataset(sample, 'data.data')

# mean = np.array([[-5, 0], [0, 5], [5, -5]])
# cov = np.array([[1, 1], [1, 1], [1, 1]])
# sample = normal_distribution_generate_boxmuller(mean, cov, 3, [100, 100, 100])
# save_dataset(sample, 'data.data')
#
# model = GaussianMixtureModel(maxiter=50, k=3, file_name='data.data')
model = GaussianMixtureModel(maxiter=100, k=3)
model.train()

model2 = KMeansModel(k=3)
model2.train()