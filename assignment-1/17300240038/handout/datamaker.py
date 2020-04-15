import random
import argparse
import math as M
import pickle
import numpy as np

def get_config():
	C = argparse.ArgumentParser()
	C.add_argument("--p1" , type = str , default = "0,1")
	C.add_argument("--p2" , type = str , default = "1,1")
	C.add_argument("--p3" , type = str , default = "-1,1")
	C.add_argument("--prop" , type = str , default = "0.4,0.5,0.1")
	C.add_argument("--seed" , type = int , default = 2333)
	C = C.parse_args()

	random.seed(C.seed)
	np.random.seed(C.seed)

	return C


def make_gaussian(d , miu , sigma):
	return np.random.normal(miu , sigma , d).reshape(d , 1)

def main():
	C = get_config()

	d = 8
	n = 1000
	p = [float(x) for x in C.prop.strip().split(",")]
	params = [[float(x) for x in y.strip().split(",")] for y in [C.p1 , C.p2 , C.p3]]

	data = [ [make_gaussian(d , *params[t]) , t] for t in range(3) for i in range(int(p[t]*n))]

	random.shuffle(data)

	with open("data.data" , "wb") as fil:
		pickle.dump(data , fil)

if __name__ == "__main__":
	main()
