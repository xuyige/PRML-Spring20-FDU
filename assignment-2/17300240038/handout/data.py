import torch as tc
import random

def int2list(x , padto):
	return [int(d) for d in str(x)][::-1] + [0 for _ in range(padto - len(str(x)))]

def gen_data_batch(C , batch_size , train = False):

	low = 0
	high = 10 ** (C.max_length - 1)

	if train:
		low = high // 2
	else:
		high = high // 2

	a = [ random.randint(low , high) for i in range(batch_size)]
	b = [ random.randint(low , high) for i in range(batch_size)]
	r = [ a[i] + b[i] for i in range(batch_size)]

	nums = [a , b , r]

	for i in range(3):
		nums[i] = [int2list(x , C.max_length) for x in nums[i]]
		nums[i] = tc.cuda.LongTensor(nums[i])
	return nums


