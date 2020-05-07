
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .data import prepare_batch, gen_data_batch, results_converter


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        #11个位的数字分别进行embedding
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        #num1.shape=[200,11]
        num1=self.embed_layer(num1)
        num2=self.embed_layer(num2)
        #num1.shape=[200,11,32]
        #将每一位数字embedding到32维更高的嵌入空间中
        input = torch.cat((num1, num2), 2).transpose(0,1)
        #input.shape=[11,200,64]
        #将input 按照 seq_length=num_length,batch_size=200,input_size=64的顺序进行输入
        output, _ = self.rnn(input)
        #output.shape=[11,200,64]
        logits = self.dense(output)
        #将输出数字从高维dense回低维
        #logits.shape=[11，200，10]
        logits = logits.transpose(0,1).clone()
        #logits shape=[200,11,10]
        return logits

class myAdvPTRNNModel(nn.Module):
    def __init__(self,model='RNN',layer=2):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        if model=='RNN':
            self.rnn = nn.RNN(64, 64, layer)
        elif model=='LSTM':
            self.rnn = nn.LSTM(64,64,layer)
        elif model== 'GRU':
            self.rnn = nn.GRU(64,64,layer)

        self.dense = nn.Linear(64, 10)


    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1=self.embed_layer(num1)
        num2=self.embed_layer(num2)
        input = torch.cat((num1, num2), 2).transpose(0,1)
        output, _ = self.rnn(input)
        logits = self.dense(output)
        logits = logits.transpose(0,1).clone()
        return logits


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.view(-1))


def train_one_step(model, optimizer, x, y, label):
    model.train()
    optimizer.zero_grad()
    logits = model(torch.tensor(x), torch.tensor(y),)
    loss = compute_loss(logits, torch.tensor(label))

    # compute gradient
    loss.backward()
    optimizer.step()
    return loss.item()


def train(arg, model, optimizer):
    steps=arg.epoch
    loss = 0.0
    accuracy = 0.0
    losses=[]
    accuracies=[]
    for step in range(steps):
        datas = gen_data_batch(batch_size=arg.batch_size, start=0, end=10**arg.digit_len-1)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=arg.digit_len+1)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        losses.append(loss)
        if step % 50 == 0:
            print('step', step, ': loss', loss)
        if step % 50 ==0:
            accuracy=evaluate(model,arg.test_batch_size,arg.digit_len,arg)
            accuracies.append(accuracy)
    return losses,accuracies

def evaluate(model,batch_size,digit_len,arg):
    datas = gen_data_batch(batch_size, start=10**(digit_len-1)*5+1, end=10**digit_len-1)
    Nums1, Nums2,  results = prepare_batch(*datas, maxlen=digit_len+1)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))
    return np.mean([o[0]==o[1] for o in zip(datas[2], res)])


def pt_main(arg):
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=arg.lr)
    loss,accuracy=train(arg, model, optimizer)
    evaluate(model)


def pt_adv_main(arg):
    '''
    Please finish your code here.
    '''
    print(arg)
    model = myAdvPTRNNModel(arg.model,arg.layer)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=arg.lr)
    loss,accuracy=train(arg, model, optimizer)
    return loss,accuracy

def plot_loss(losses,model='RNN',layer='3',digitlen=10,lr=1e-3):
    print(len(losses))
    x=np.arange(0,len(losses))
    y=losses
    label='%s-loss' %str(model)
    #label='%s-loss' %str(lr)
    plt.plot(x,y,label=label)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.legend()
    savepath=label+'.png'
    plt.savefig(savepath)


def plot_accuracy(accuracies,model='RNN',layer='3',digitlen=10,lr=1e-3):
    print(len(accuracies))
    x=np.arange(0,len(accuracies)*50,50)
    y=accuracies
    label='%s-accuracy' %str(model)
    #label='%s-accuracy' %str(lr)
    plt.plot(x,y,label=label)
    plt.xlabel('Iteration number')
    plt.ylabel('Accuracy value')
    plt.legend()
    savepath=label+'.png'
    plt.savefig(savepath)

def test1 (arg):
    losses=[]
    accuracies=[]
    models=[]
    layers=[]
    for i in range(1,5):
        models.append(arg.model)
        arg.layer=i
        layers.append(i)
        loss,accuracy=pt_adv_main(arg)
        losses.append(loss)
        accuracies.append(accuracy)
    i=0
    for loss in losses:
       plot_loss(loss,models[i],layers[i])
       i+=1
    plt.clf()
    i=0
    for accuracy in accuracies:
       plot_accuracy(accuracy,model=models[i],layer=layers[i])
       i+=1

def test2 (arg):
    digitlen=[]
    losses=[]
    accuracies=[]
    for i in [10,100,1000,5000]:
        arg.digit_len=i
        digitlen.append(i)
        loss,accuracy=pt_adv_main(arg)
        losses.append(loss)
        accuracies.append(accuracy)
    i=0
    for loss in losses:
       plot_loss(loss,digitlen=digitlen[i])
       i+=1
    plt.clf()
    i=0
    for accuracy in accuracies:
       plot_accuracy(accuracy,digitlen=digitlen[i])
       i+=1

def test3(arg):
    models=['RNN','GRU','LSTM']
    losses=[]
    accuracies=[]

    for model in models:
        arg.model=model
        loss,accuracy=pt_adv_main(arg)
        losses.append(loss)
        accuracies.append(accuracy)
    i=0
    for loss in losses:
       plot_loss(loss,model=models[i])
       i+=1
    plt.clf()
    i=0
    for accuracy in accuracies:
       plot_accuracy(accuracy,model=models[i])
       i+=1

def test4(arg):
    losses=[]
    accuracies=[]
    lrs=[1e-5,1e-4,1e-3,1e-2,1e-1]
    for lr in lrs:
        arg.lr=lr
        loss,accuracy=pt_adv_main(arg)
        losses.append(loss)
        accuracies.append(accuracy)
    i=0
    for loss in losses:
       plot_loss(loss,lr=lrs[i])
       i+=1
    plt.clf()
    i=0
    for accuracy in accuracies:
       plot_accuracy(accuracy,lr=lrs[i])
       i+=1

