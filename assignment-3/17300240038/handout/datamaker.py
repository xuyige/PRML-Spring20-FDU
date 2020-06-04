import random
import argparse
import math as M
import pickle
import numpy as np

mus = [ [0.1,0.1] , [3,4] , [0.5,3] ]
sigmas = [
	[[0.9 , 0.3] ,
	[0.3 , 0.7]] ,

	[[0.8 , 0.2] ,
	[0.2 , 0.7]] ,

	[[0.8 , 0.3] ,
	[0.3 , 0.8]] ,
]


def make_gaussian(mu , sigma , label):
	return [np.random.multivariate_normal(mu , sigma) , label]

def main():

	n = 1000
	p = [0.3 , 0.3 , 0.4]

	data = [ make_gaussian(mus[i] , sigmas[i] , i) for i in range(3) for x in range(int(n*p[i])) ]

	random.shuffle(data)

	with open("data.data" , "wb") as fil:
		pickle.dump(data , fil)

if __name__ == "__main__":
	main()
