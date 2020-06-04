import numpy as np
from tqdm import tqdm
import pdb

def N(x , mu , sigma):
	'''x在N(mu,sigma)下的概率密度
	
	x: (n , 2)
	mu: (2)
	sigma: (2,2)
	'''
	n = len(x)

	a = float((np.linalg.det(sigma) * (6.28 ** 2)) ** -0.5)
	
	b = (x - mu.reshape(1,2)).reshape(n , 2,1) #(n , 2,1)
	c = np.linalg.inv(sigma)				   #(2,2)
	d = np.exp(-0.5 * ( np.matmul(np.matmul(b.transpose(0,2,1) , c) , b)) ).reshape(n)

	return a * d

def run_GMM(xs , K):
	'''
		xs: [n , 2]
	'''
	n 		= len(xs)
	
	ps 		= np.ones([K]) / K
	mus 	= np.random.randn(K , 2)
	sigmas 	= np.array([np.eye(2 , 2) for k in range(K)])

	for i in tqdm(range(1000) , ncols = 120):
		#pdb.set_trace()
		#E -step
		phi 	= np.array( [N(xs , mus[k] , sigmas[k]) for k in range(K)] ) # (K,n)
		gama 	= ps.reshape(K,1) * phi # (K,n)
		gama 	= gama / gama.sum(0).reshape(1,n) #(K,n)
		
		#M-step
		mus 	= np.matmul(gama , xs) / gama.sum(1).reshape(K,1)

		a 		= (xs.reshape(1,n,2) - mus.reshape(K,1,2)).reshape(K,n,2,1) # (K,n,2,1)
		a 		= np.matmul(a , a.transpose(0,1,3,2)) # (K,n,2,2)
		sigmas 	= (gama.reshape(K,n,1,1) * a).sum(1) #(K,2,2)
		sigmas 	= sigmas / gama.sum(1).reshape(K,1,1)

		ps 		= gama.sum(1) / n

	return ps , mus , sigmas

def pred_GMM(xs , K , ps , mus , sigmas , ret_p = False):
	phi = np.array( [N(xs , mus[k] , sigmas[k]) for k in range(K)] ) # (K,n)
	if ret_p:
		return phi.transpose(1,0) #(n,K)

	return phi.argmax(0) #(n)
