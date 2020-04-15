import random
import argparse
import math as M
import pickle
from .model import DiscriminativeModel , GenerativeModel
import numpy as np
from .utils import Logger
from tqdm import tqdm
import pdb

def get_config():
	C = argparse.ArgumentParser()
	C.add_argument("--data" , type = str , default = "data.data")
	C.add_argument("--log"  , type = str , default = "log.log")
	C.add_argument("--seed" , type = int , default = 2333)


	C.add_argument("--train_step"	, type = int , default = 10000)
	C.add_argument("--dev_prop"  	, type = float , default = 0.1)
	C.add_argument("--test_prop" 	, type = float , default = 0.1)
	C.add_argument("--lr" 		 	, type = float , default = 1e-2)
	C.add_argument("--bs" 		 	, type = int , default = 1)

	C.add_argument("--d" 			, type = int , default = 8)
	C = C.parse_args()

	random.seed(C.seed)
	np.random.seed(C.seed)

	C.logger = Logger(C.log)
	C.log = C.logger.log
	C.log_no_print = C.logger.log_no_print

	return C

def load_data(C):
	with open(C.data , "rb") as fil:
		data = pickle.load(fil)

	d_l , t_l = int(C.dev_prop * len(data)) , int(C.test_prop * len(data))
	dev_data , test_data , train_data = data[:d_l] , data[d_l : d_l+t_l] , data[d_l+t_l : ]

	C.label_num = max([x[1] for x in data]) + 1
	C.label_dist = [[x[1] for x in train_data].count(l) / len(train_data) for l in range(C.label_num)]

	C.log (" # labels : %d" % C.label_num)
	for data , name in zip([dev_data , test_data , train_data] , ["dev  " , "test " , "train"]):
		C.log ("# %s samples : %d" % (name , len(data)))
	for data , name in zip([dev_data , test_data , train_data] , ["dev  " , "test " , "train"]):
		C.log ("label dist. in %s : %s" % (name , str([[x[1] for x in data].count(l) for l in range(C.label_num)])))
	#pdb.set_trace()

	return (dev_data , test_data , train_data) , C.label_num

def process(C , model , batch , train = False):

	loss = 0
	ghit = 0
	grad = 0
	for sample in batch:
		x , label = sample

		p = model.forward(x)

		loss += -np.log(p[label])
		grad += model.backward(label)

		if not train:
			for i in range(C.label_num):
				p = p / C.label_dist[i]

		ghit += int(np.argmax(p) == label)

	if train:
		grad = grad / len(batch)
		model.W -= grad * C.lr

	return loss / len(batch) , ghit / len(batch)


def run(C , dev_step = 100 , mode = "discriminative"):
	R = C.R

	(dev_data , test_data , train_data) , label_num = load_data(C)

	model = {
		"discriminative" : DiscriminativeModel(C.d , label_num) , 
		"generative"     : GenerativeModel    (C.d , label_num ) , 
	}[mode]

	C.logger.add_line()
	C.log("Training started!\n")

	pbar = tqdm(range(C.train_step) , ncols = 100)
	train_losss = []
	train_ghits = []
	dev_loss = 0
	dev_acc  = 0
	best_dev_acc = -1
	best_dev_step = -1
	best_dev_W = None
	for step in pbar:
		pbar.set_description_str("Step %d" % step)

		batch = random.sample(train_data , C.bs)

		loss , acc = process(C , model , batch , train = True)
		train_losss += [loss]
		train_ghits += [acc]
		while len(train_losss) > 50:
			train_losss.pop(0)
			train_ghits.pop(0)

		if step > 0 and (step % dev_step == 0):
			dev_loss , dev_acc = process(C , model , dev_data , train = False)

		if best_dev_acc < 0 or best_dev_acc < dev_acc:
			best_dev_acc = dev_acc
			best_dev_step = step
			best_dev_W = model.W

		postfix = "Train loss = %.4f , Train acc = %.2f , Dev loss = %.4f , Dev acc = %.2f" % (
			sum(train_losss) / len(train_losss) , 
			sum(train_ghits) / len(train_ghits) , 
			dev_loss , 
			dev_acc , 
		)
		pbar.set_postfix_str(postfix)
		if step % dev_step == 0:
			C.log_no_print(("At step %d: " % step) + postfix)

	C.logger.add_line()
	C.log("Got best dev result at step %d : acc = %.2f" % (best_dev_step , best_dev_acc))
	R.log("Report for %s model:" % mode)
	R.log("Got best dev result at step %d : acc = %.2f%%" % (best_dev_step , best_dev_acc * 100))

	model.W = best_dev_W
	C.log("Reloaded best model parameter.")

	test_loss , test_acc = process(C , model , test_data , train = False)
	C.log("Test accurancy is %.2f" % test_acc)
	R.log("Test accurancy is %.2f%%" % (test_acc * 100))

	C.logger.add_line()
	R.add_line()

def main():
	C = get_config()
	C.R = Logger("report.log" , False , False)


	run(C ,  mode = "discriminative")
	run(C ,  mode = "generative")


	C.logger.close()
	C.R.close()

if __name__ == "__main__":
	main()









