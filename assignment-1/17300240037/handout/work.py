import numpy as np
from matplotlib import pyplot as plt 
from .MakeGaussian import makedata
from .models import DiscriminativeModel, GenerativeModel
def loadData():
	'''
	生成数据后，以(X_train, y_train, X_test, y_test)的格式返回给调用的函数
	'''
	num1 = 700
	num2 = 1100
	num3 = 1200
	data = makedata(num1, num2, num3)
	num = num1 + num2 + num3
	nTrain = int(num * 0.9)
	nTest = num - nTrain
	'''
	np.set_printoptions(threshold=np.inf)
	f = open('./gauss.data', 'w')
	f.write(str(data))
	'''
	X = data[ : , : 2]
	y = data[ : , 2 :]
	X_train = X[ :nTrain]
	X_test = X[nTrain : ]
	y_train = y[ :nTrain]
	y_test = y[nTrain : ]
	y_train = (y_train - 0.5).astype(int).reshape((nTrain, ))
	y_test = (y_test - 0.5).astype(int).reshape((nTest, ))

	return X_train, y_train, X_test, y_test

def main():
	X_train, y_train, X_test, y_test = loadData()
	'''
	for i in range(X_test.shape[0]):
		if(y_test[i] == 0):
			plt.plot(X_test[i, 0], X_test[i, 1], 'ob')
		if(y_test[i] == 1):
			plt.plot(X_test[i, 0], X_test[i, 1], 'or')
		if(y_test[i] == 2):
			plt.plot(X_test[i, 0], X_test[i, 1], 'og')
	plt.show()
	'''
	model = DiscriminativeModel()
	'''
	model.W = np.array([[-2, 0, 2], [0, 0, 0], [0, 10, 0]], dtype = 'float64')

	y_pred = model.predict(X_test)
	test_accuracy = np.mean(y_test == y_pred)

	for i in range(X_test.shape[0]):
		if(y_test[i] != y_pred[i]):
			print("y_test = %d, y_pred = %d, X_test = %s\n" % (y_test[i], y_pred[i], str(X_test[i])) )
	print(test_accuracy)
	'''
	history = model.train(X_train, y_train)
	y_pred = model.predict(X_test)
	test_accuracy = np.mean(y_test == y_pred)
	print("The accuracy of the Discriminative Model = %f\n" % test_accuracy)

	model = GenerativeModel()
	model.train(X_train, y_train)
	y_pred = model.predict(X_test)
	test_accuracy = np.mean(y_test == y_pred)
	print("The accuracy of the Generative Model = %f\n" % test_accuracy)

if __name__ == '__main__':
	main()