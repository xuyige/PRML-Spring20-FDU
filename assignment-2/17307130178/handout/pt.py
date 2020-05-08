
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy.random import uniform
from .data import prepare_batch, gen_data_batch, results_converter


# Part I

# In this part, you are required to learn how to write your source code based on 
# Tensorflow 2.0 (TF) or PyTorch (PT). Source code based on TF or PT is available 
# in directory [`example`](https://github.com/xuyige/PRML-Spring20-FDU/blob/master/assignment-2/example) .
# You are required to clone one of them to your own directory and finish the rest of codes. 
# (If you find any mistakes from the example source code, please contact TA 
# [@xuyige](https://github.com/xuyige).)

class myPTRNNModel(nn.Module):
    def __init__(self,layers=2):
        super().__init__()
        self.loss_history = []
        self.accuracy = []
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, layers)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        # num1 (200,11,32)
        # input (11,200,64)
        # output (11,200,64)
        # logits (200,11,10)

        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)
        input = torch.cat((num1, num2), -1).transpose(0,1)
        # print(input.shape)
        output, hidden = self.rnn(input)
        logits = self.dense(output)
        logits = logits.transpose(0,1).clone()
        # print(input.shape, output.shape ,logits.shape)
        return logits
    



class myAdvPTRNNModel(nn.Module):
    def __init__(self,layers=2,model="GRU"):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.loss_history = []
        self.accuracy = []
        self.embed_layer = nn.Embedding(10, 32)
        if model=="GRU":
            self.rnn = nn.GRU(64,64,layers)
        elif model=="LSTM":
            self.rnn = nn.LSTM(64,64,layers)
        else:
            self.rnn = nn.RNN(64,64,layers)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        # num1 (200,11,32)
        # input (11,200,64)
        # output (11,200,64)
        # logits (200,11,10)
        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)
        input = torch.cat((num1, num2), -1).transpose(0,1)
        # print(input.shape)
        output, hidden = self.rnn(input)
        logits = self.dense(output)
        logits = logits.transpose(0,1).clone()
        return logits


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.view(-1))


def train_one_step(model, optimizer, x, y, label, device='cpu'):
    model.train()
    optimizer.zero_grad()
    num1 = torch.LongTensor(x).to(device)
    num2 = torch.LongTensor(y).to(device)
    logits = model(num1, num2)
    res = torch.LongTensor(label).to(device)
    loss = compute_loss(logits, res)

    # compute gradient
    loss.backward()
    optimizer.step()
    return loss.item()


def train(steps, model, optimizer, device='cpu', digitlen=10):
    loss = 0.0
    accuracy = 0.0
    end_num = 5*10**(digitlen-1)
    for step in range(steps):
        datas = gen_data_batch(batch_size=100, start=0, end=end_num)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=digitlen+1)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results, device)
        model.loss_history.append(loss)
        if step % 10 == 0:
            accuracy = evaluate(model,device,digitlen)
            model.accuracy.append(accuracy)

        if step % 50 == 0:
            print('step', step, ': loss', loss ,':accuracy', accuracy)
            
    return loss


def evaluate(model, device='cpu', digitlen=10):
    start_num = 5*10**(digitlen-1)+1
    end_num = 10**digitlen - 1
    datas = gen_data_batch(batch_size=100, start=start_num, end=end_num)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=digitlen+1)
    with torch.no_grad():
        logits = model(torch.LongTensor(Nums1).to(device), torch.LongTensor(Nums2).to(device))
        logits = logits.cpu() #转回cpu
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    accuracy = np.mean([o[0]==o[1] for o in zip(datas[2], res)])
    # print('accuracy is: %g' % accuracy)
    return accuracy

def loss_draw(model):
    plt.subplot(211)
    plt.plot(model.loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')

    plt.subplot(212)
    plt.plot(model.accuracy)
    plt.xlabel('Iteration number / 10')
    plt.ylabel('Accuracy')
    plt.show()



def pt_main():
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    digitnum = 100

    model = myPTRNNModel().to(cuda0)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, cuda0, digitnum)
    loss_draw(model)


def pt_adv_main():
    '''
    Please finish your code here.
    '''
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    digitnum = 100

    model = myAdvPTRNNModel(3,"RNN").to(cuda0)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    train(500, model, optimizer, cuda0, digitnum)
    loss_draw(model)



# 测试不同RNN层数
def test1():
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    digitnum = 10
    loss, accuracy = [] , []

    for i in range(1,5):
        model = myAdvPTRNNModel(i).to(cuda0)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        train(500, model, optimizer, cuda0, digitnum)
        
        loss.append(model.loss_history)
        accuracy.append(model.accuracy)


    plt.subplot(211)
    for i in range(len(loss)):
        x = np.arange(len(loss[i]))
        plt.plot(x,loss[i], label="RNN-"+str(i+1))
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.legend()

    plt.subplot(212)
    for i in range(len(accuracy)):
        x = np.arange(len(accuracy[i]))
        plt.plot(x,accuracy[i], label="RNN-"+str(i+1))
    plt.xlabel('Iteration number / 10')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# 测试GRU/LSTM
def test2():
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    digitnum = 100
    loss, accuracy = [] , []

    model1 = myPTRNNModel(3).to(cuda0)
    optimizer = torch.optim.Adam(params=model1.parameters(), lr=0.001)
    train(500, model1, optimizer, cuda0, digitnum)
    loss.append(model1.loss_history)
    accuracy.append(model1.accuracy)

    model2 = myAdvPTRNNModel(3,"GRU").to(cuda0)
    optimizer = torch.optim.Adam(params=model2.parameters(), lr=0.001)
    train(500, model2, optimizer, cuda0, digitnum)
    loss.append(model2.loss_history)
    accuracy.append(model2.accuracy)

    model3 = myAdvPTRNNModel(3,"LSTM").to(cuda0)
    optimizer = torch.optim.Adam(params=model3.parameters(), lr=0.001)
    train(500, model3, optimizer, cuda0, digitnum)
    loss.append(model3.loss_history)
    accuracy.append(model3.accuracy)

    plt.subplot(211)
    x = np.arange(len(loss[0]))
    plt.plot(x,loss[0], label="RNN")
    plt.plot(x,loss[1], label="GRU")
    plt.plot(x,loss[2], label="LSTM")
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.legend()

    plt.subplot(212)
    x = np.arange(len(accuracy[0]))
    plt.plot(x,accuracy[0], label="RNN")
    plt.plot(x,accuracy[1], label="GRU")
    plt.plot(x,accuracy[2], label="LSTM")
    plt.xlabel('Iteration number / 10')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# 测试学习率的影响
def test3():
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    digitnum = 10
    loss, accuracy, learning_rate = [] , [] , []

    for i in range(5):
        model = myPTRNNModel(3).to(cuda0)
        lr = 10**uniform(-2.2, -1.8)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        train(400, model, optimizer, cuda0, digitnum)
        
        loss.append(model.loss_history)
        accuracy.append(model.accuracy)
        learning_rate.append(lr)


    plt.subplot(211)
    for i in range(len(loss)):
        x = np.arange(len(loss[i]))
        plt.plot(x,loss[i], label="RNN-"+str(learning_rate[i]))
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.legend()

    plt.subplot(212)
    for i in range(len(accuracy)):
        x = np.arange(len(accuracy[i]))
        plt.plot(x,accuracy[i], label="RNN-"+str(learning_rate[i]))
    plt.xlabel('Iteration number / 10')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# pt_adv_main()