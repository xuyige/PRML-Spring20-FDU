from handout import *
import numpy as np

K = 3
DIM = 2

gen_data(means=np.array([[5 * np.cos(2*i*np.pi/K), 5 * np.sin(2*i*np.pi/K)] for i in range(K)]),
         covs=np.array([np.eye(DIM) * 0.3 * np.random.rand()] * K),
         scale=[400] * K)
data = load_data()

gmm = GMM(k=K, dim=DIM, max_iter=100)
gmm.fit(data, pre_train=True, plot=True)
