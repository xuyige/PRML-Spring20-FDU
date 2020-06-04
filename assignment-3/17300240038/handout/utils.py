import matplotlib.pyplot as plt
import pdb
def watch_data(xys , labs = None):
	colors = [
		[0,0,1] , 
		[0,1,0] , 
		[1,0,0] , 
	]
	for i in range(len(xys)):
		if labs is not None:
			lab = labs[i]
		plt.scatter(xys[i][0] , xys[i][1] , s = 5 , color = colors[lab])

	plt.show()

def mode(a):
	return float(a.sum()) ** 2 

def match(a , b):
	'''根据与a的距离将b重排序

	ret[i] = j：a[i] -> b[j]
	'''
	n = len(a)
	ret = [None for _ in range(n)]
	idxs = list(range(n))
	for i in range(n):
		best_idx = -1
		#pdb.set_trace()
		for j in range(len(idxs)):
			if best_idx < 0 or mode(b[idxs[j]] - a[i]) < mode(b[idxs[best_idx]] - a[i]):
				best_idx = j
		ret[i] = idxs[best_idx]
		idxs.pop(best_idx)
	return ret

