import random
import argparse
import math as M
import pickle
import numpy as np
from tqdm import tqdm
import pdb
from .utils import *
from .GMM import *

def get_config():
	C = argparse.ArgumentParser()
	C.add_argument("--data" , type = str , default = "data.data")
	C.add_argument("--seed" , type = int , default = 2333)

	C = C.parse_args()

	random.seed(C.seed)
	np.random.seed(C.seed)
	return C

def load_data(C):
	with open(C.data , "rb") as fil:
		data = pickle.load(fil)

	return data


def main():
	C 		= get_config()
	data 	= load_data(C)
	ora_mus = np.array([ [0.1,0.1] , [3,4] , [0.5,3] ])

	xs 		= np.array([d[0] for d in data])
	labs 	= np.array([d[1] for d in data])
	K 		= max(labs) + 1
	n 		= len(xs)

	watch_data([d[0] for d in data] , labs)

	ps , mus , sigmas = run_GMM (xs , K)
	prd_lab 		  = pred_GMM(xs , K , ps , mus , sigmas)

	permute = match(mus , ora_mus)

	for i in range(len(labs)):
		prd_lab[i] = permute[prd_lab[i]]

	watch_data([d[0] for d in data] , prd_lab)

	bingo = (labs == prd_lab).sum()

	print (ps)
	print (mus)
	print (sigmas)

	print ("%d / %d = %.4f" % (bingo , n , bingo / n))

if __name__ == "__main__":
	main()



