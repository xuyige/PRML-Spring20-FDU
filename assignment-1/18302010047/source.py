'''
    PRML Assignment 1
    Spring 2020, Fudan University
    18302010047 Dai Yuchun

    you can run the code by the following command:
    
    Data generation(save to a.data as default):
        
        python source.py generate
    
    Run generative model(make sure that there is a data file):

        python source.py generative_model
    
    Run discriminative model(make sure that there is a data file):

        python source.py discriminative_model
'''

#package import
import sys
import math
import argparse
import numpy as np
from matplotlib import pyplot as plt


#data generation
def data_generate(args):
    print('data generate\n')

    distributions = [[float(x) for x in label.strip('][').split(',')] \
                                        for label in [args.d1, args.d2, args.d3]]
    plabel = [float(x) for x in args.plabel.strip('][').split(',')]
    size = args.size
    dim = args.dim
    data = [
            [np.random.normal(distributions[label][0], \
                    distributions[label][1], dim).reshape(dim,1), label]
            for label in range(3) \
            for i in range(int(size*plabel[label]))
    ]
    np.random.shuffle(data)
    with open("a.data", "w") as f:
        for example in data:
            f.write(" ".join([str(*example[0][i]) for i in range(dim)])+" "+str(example[1])+"\n")              

    # data visualization

    # def get_axis(items):
    #     x = [float(example[0][0]) for example in items]
    #     y = [float(example[0][1]) for example in items]
    #     return x, y

    # plt.scatter(*get_axis(list(filter(lambda x: x[1] == 0, data))), color='b', label='A', alpha=0.3)
    # plt.scatter(*get_axis(list(filter(lambda x: x[1] == 1, data))), color='r', label='B', alpha=0.3)
    # plt.scatter(*get_axis(list(filter(lambda x: x[1] == 2, data))), color='g', label='C', alpha=0.3)
    # plt.legend()
    # plt.show()

#generative model
class GenerativeModel():
    def __init__(self, dim):
        self.labelNum = 3
        self.dim = dim
        self.mu = np.zeros((self.labelNum, self.dim))
        self.delta = np.zeros((self.labelNum, self.dim))
        self.count = np.zeros(self.labelNum)
    
    def train(self, data):
        n = len(data)
        for i in range(3):
            example = list(filter(lambda x: x[1]==i, data))
            self.count[i] = len(example)
            for j in range(self.dim):
                self.mu[i][j] = sum([float(x[0][j]) for x in example]) / self.count[i]
                self.delta[i][j] = sum([(float(x[0][j]) - self.mu[i][j]) ** 2 for x in example]) / self.count[i]

    def predict(self, data, save_path):
        n = len(data)
        app = np.zeros(self.labelNum)
        prior = self.count / sum(self.count)
        pre_labels = []
        for example in data:
            probs = [math.exp(-0.5*sum([(example[0][j] - self.mu[i][j]) ** 2 \
                                        for j in range(self.dim)]))/2.0/np.pi*prior[i] \
                                             for i in range(3)]
            label = np.argmax(np.array(probs))
            app[label] += 1
            pre_labels.append(label)
        
        acc = np.zeros(self.labelNum+1)
        for i in range(len(pre_labels)):
            if(pre_labels[i] == data[i][1]):
                acc[pre_labels[i]]+=1
                acc[self.labelNum]+=1
        
        with open(save_path, "a") as f:
            f.write('Total Accuracy: '+str(acc[self.labelNum]/n)+'\n')
            print('Total Accuracy: '+str(acc[self.labelNum]/n))
            for i in range(3):
                f.write('Accuracy label '+str(i)+' '+str(acc[i]/app[i])+'\n')
                print('Accuracy label '+str(i)+' '+str(acc[i]/app[i]))


#generative model main function
def generative_model(args):
    print('generative model\n')

    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            items = list(map(float, items))
            data.append([items[:-1], int(items[-1])])
    
    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]
    model = GenerativeModel(dim=args.dim)
    model.train(train_data)
    print('Train set:')
    model.predict(train_data, args.save_path)
    print('Test set:')
    model.predict(test_data, args.save_path)


#discriminative model
class DiscriminativeModel():
    def __init__(self, dim):
        self.labelNum = 3
        self.dim = dim
        self.W = np.random.rand(self.labelNum, self.dim+1)

    def train(self, data, batch=64, epochs=100, lr=1e-2, save_path='dm_result.txt'):
        n = len(data)

        batchNum = n // batch
        if n % batch != 0:
             batchNum += 1

        onehot = np.zeros((n, self.labelNum, 1))
        for i in range(n):
            onehot[i][data[i][1]][0] = 1

        x = np.array([example[0] for example in data])
        x = np.insert(x, self.dim, 1, axis=1)
        x = np.array([np.reshape(xx, (-1, 1)) for xx in x])
        y = np.array([example[1] for example in data])

        # loss_p = []
        # acc_p = []
        with open(save_path, 'w') as f:
            for epoch in range(epochs):
                loss = 0
                ok = 0
                for now in range(batchNum):
                    delta = np.zeros((self.labelNum, self.dim+1))
                    x_ = x[now*batch:min(n, (now+1)*batch)]
                    one = onehot[now*batch:min(n, (now+1)*batch)]
                    y_t = y[now*batch:min(n, (now+1)*batch)]
                    y_ = np.matmul(self.W, x_)
                    for i in range(y_.shape[0]):
                        y_[i] = np.exp(y_[i])/sum(np.exp(y_[i]))
                        loss -= np.log(y_[i][y_t[i]])                    
                        if(int(y_t[i]) == int(np.argmax(y_[i]))):
                            ok += 1
                        y_[i] -= one[i]
                        delta += np.matmul(y_[i], np.transpose(x_[i]))

                    self.W -= delta / batch * lr
                # loss_p.append(float(loss))
                # acc_p.append(float(ok)/n)
                f.write(f'train epoch {epoch}: loss: {loss} Accuracy:{float(ok)/n}\n')
                print(f'train epoch {epoch}: loss: {loss} Accuracy:{float(ok)/n}')
    
        # train visualization
        # x = list(range(epochs))
        # plt.figure('accuracy rate')
        # ax = plt.gca()
        # ax.set_xlabel('epochs')
        # ax.set_ylabel('accuracy rate')
        # ax.plot(x, acc_p, color='b',linewidth=1, alpha=0.6)
        # plt.show()


    def predict(self, data, save_path='gm_result.txt'):
        n = len(data)
        x = np.array([example[0] for example in data])
        x = np.insert(x, self.dim, 1, axis=1)
        x = np.array([np.reshape(xx, (-1, 1)) for xx in x])
        y = np.array([example[1] for example in data])

        ok = 0
        y_ = np.matmul(self.W, x)
        for i in range(y_.shape[0]):
            y_[i] = np.exp(y_[i])/sum(np.exp(y_[i]))
            if(int(y[i]) == int(np.argmax(y_[i]))):
                ok += 1
        
        with open(save_path, 'a') as f:
            f.write(f'test accuracy:{float(ok)/n}\n')
            print(f'test accuracy:{float(ok)/n}')

#discriminative model main function
def discriminative_model(args):
    print('discriminative model\n')
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            items = list(map(float, items))
            data.append([items[:-1], int(items[-1])])
    
    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]
    model = DiscriminativeModel(args.dim)
    model.train(train_data, epochs=args.epochs, lr=args.lr, save_path=args.save_path)
    model.predict(test_data, save_path=args.save_path)

# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PRML assignment1.')
    subparser = parser.add_subparsers(help='some subfunctions')

    gd = subparser.add_parser('generate', help='generate data')
    gd.add_argument('--d1', type=str, default="[0,1]")
    gd.add_argument('--d2', type=str, default="[-4,1]")
    gd.add_argument('--d3', type=str, default="[4,1]")
    gd.add_argument('--size', type=int, default=10000)
    gd.add_argument('--plabel', type=str, default="[0.4,0.3,0.3]")
    gd.add_argument('--dim', type=int, default=2)
    gd.set_defaults(func=data_generate)

    gm = subparser.add_parser('generative_model', help='generative model')
    gm.add_argument('--data_path', type=str, default='a.data')
    gm.add_argument('--save_path', type=str, default='gm_result.txt')
    gm.add_argument('--dim', type=int, default=2)
    gm.set_defaults(func=generative_model)

    dm = subparser.add_parser('discriminative_model', help='discirminative model')
    dm.add_argument('--data_path', type=str, default='a.data')
    dm.add_argument('--save_path', type=str, default='dm_result.txt')
    dm.add_argument('--epochs', type=int, default=100)
    dm.add_argument('--lr', type=float, default=1e-2)
    dm.add_argument('--dim', type=int, default=2)
    dm.set_defaults(func=discriminative_model)

    args = parser.parse_args(sys.argv[1:])
    args.func(args)
