import numpy as np
import torch
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .data import gen_data_batch
import pdb

class myPTRNNModel(nn.Module):
	def __init__(self , d = 32):
		super().__init__()
		self.emb = nn.Embedding(10, d)

		self.rnn = nn.RNN(input_size = 2*d , hidden_size = 2*d , batch_first = True)
		self.lno = nn.Linear(2*d , 10)

	def forward(self, num1, num2):

		num1 = num1.cuda()
		num2 = num2.cuda()

		x = self.emb(num1)
		y = self.emb(num2)
		
		r = tc.cat([x,y] , dim = -1)

		r , h = self.rnn(r)
		r = F.relu(r)

		r = self.lno(r)

		return r


class myAdvPTRNNModel(nn.Module):
	def __init__(self , d = 32):
		super().__init__()
		self.emb = nn.Embedding(10, d)

		self.ln1 = nn.Linear(4*d , 2*d)
		self.ln2 = nn.Linear(2*d , 10)

	def forward(self, num1, num2):

		num1 = num1.cuda()
		num2 = num2.cuda()

		x = self.emb(num1)
		y = self.emb(num2)

		bs , n , d = x.size()
		
		r = tc.cat([x,y] , dim = -1)


		q = tc.cat([r.new_zeros(bs , 1 , 2*d) , r[:,:-1,:] ] , dim = 1)
		r = tc.cat([r , q] , dim = -1)


		r = F.relu(self.ln1(r))
		r = self.ln2(r)

		return r


def compute_loss(logits, labels):
	losses = nn.CrossEntropyLoss()
	return losses(logits.view(-1 , 10), labels.cuda().view(-1))
	#return losses(logits, labels)


def train_one_step(model, optimizer, x, y, label):
	model.train()
	optimizer.zero_grad()
	logits = model(x , y)
	loss = compute_loss(logits, label)

	# compute gradient
	loss.backward()
	optimizer.step()
	return loss.item()


def train(C , steps, model, optimizer):
	loss = 0.0

	losses = []

	pbar = tqdm(range(steps) , ncols = 100 , desc = "Training")
	for step in pbar:
		a, b, r = gen_data_batch(C , C.batch_size , train = True)
		loss = train_one_step(model , optimizer , a , b , r)

		losses.append(float(loss))
		while len(losses) > 50:
			losses.pop(0)
		if step % 50 == 0:
			C.logger.log("step %d: loss %.4f" % (step , loss))
		pbar.set_postfix_str("Train loss: %.4f" % (sum(losses) / len(losses)))

	C.logger.log("\n")

	return loss


def evaluate(C , model):

	good_hit = 0
	good_bit = 0
	for i in tqdm(range(C.eval_size) , ncols = 100 , desc = "Testing"):
		a , b , r = gen_data_batch(C , 1 , train = False)
		with torch.no_grad():
			pred = model(a, b)[0]
		pred = tc.max(pred , -1)[1]
		r = r[0]

		good_bit += int((r == pred).long().sum())
		good_hit += int(bool((r == pred).all()))

	acc = "Accurancy : %d / %d = %.2f%%" % (good_hit , C.eval_size , good_hit / C.eval_size * 100)
	C.logger.log(acc)
	C.R.log(acc)

	acc = "Per-bit Accurancy : %d / %d = %.2f%%" % (
		good_bit , 
		C.eval_size * C.max_length , 
		good_bit / (C.eval_size * C.max_length) * 100
	)
	C.logger.log(acc)
	C.R.log(acc)


def pt_main(C , model_class = myPTRNNModel):
	model = model_class(d = C.d).cuda()
	optimizer = torch.optim.Adam(model.parameters() , 0.001)
	train(C , 1000, model, optimizer)
	evaluate(C , model)


def pt_adv_main(C):
	pt_main(C , myAdvPTRNNModel)
