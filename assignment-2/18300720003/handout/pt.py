import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pylab as pl
from .data import prepare_batch, gen_data_batch, results_converter

class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2,batch_first = True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)
        input = torch.cat((num1, num2), 2)
        r_out, (h_n) = self.rnn(input, None)
        logits = self.dense(r_out)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self,layers=2,model="lstm"):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        if model=="gru":
            self.rnn = nn.GRU(64,64,layers,batch_first = True)
        elif model=="lstm":
            self.rnn = nn.LSTM(64,64,layers,batch_first = True)
        else:
            self.rnn = nn.RNN(64,64,layers,batch_first = True)
        self.dense = nn.Linear(64, 10)
    def forward(self, num1, num2):
        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)
        input = torch.cat((num1, num2), 2)
        r_out, (h_n) = self.rnn(input)
        logits = self.dense(r_out)
        return logits


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.view(-1))


def train_one_step(model, optimizer, x, y, label):
    model.train()
    optimizer.zero_grad()
    logits = model(torch.tensor(x), torch.tensor(y))
    loss = compute_loss(logits, torch.tensor(label))

    #compute gradient
    loss.backward()
    optimizer.step()
    return loss.item(),logits

Loss_list = []
Accuracu_train=[]
Accuracy_list_test = []
def train(steps, model, optimizer,max_digit=10,max_digit_test=10):
    loss = 0.0
    accuracy = 0.0
    global Loss_list
    global Accuracy_list
    for step in range(steps):
        datas = gen_data_batch(batch_size=30, start=0, end=10**float(max_digit)-1)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=max_digit+1)
        loss,logits = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        if step % 50 == 0:
            print('step', step, ': loss', loss)
            Loss_list.append(loss)
            logits = logits.detach().numpy()
            pred = np.argmax(logits, axis=-1)
                
            res = results_converter(pred)
           
            Accuracu_train.append(np.mean([o[0]==o[1] for o in zip(datas[2], res)]))
            Accuracy_list_test.append(evaluate(model,max_test_digit=max_digit_test))
    return loss

def evaluate(model,max_test_digit=1000):
    datas = gen_data_batch(batch_size=200, start=0, end=10**float(max_test_digit)-1)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=max_test_digit+1)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    #for o in list(zip(datas[2], res))[:20]:
     #    print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))
    return np.mean([o[0]==o[1] for o in zip(datas[2], res)])
    

def pt_main(train_length=100,test_length=100):
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-04, amsgrad=False)
    train(3000, model, optimizer,max_digit=train_length,max_digit_test=test_length)
    evaluate(model,max_test_digit=test_length)
    plt.subplot(211)
    plt.plot(Loss_list)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.subplot(212)
    plt.plot(Accuracy_list)
    plt.xlabel('Iteration number / 10')
    plt.ylabel('Accuracy')
    plt.show()


def pt_adv_main(layer=2,step=3000,rnn="rnn",train_length=100,test_length=100):
    model = myAdvPTRNNModel(layer,rnn)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-04, amsgrad=False)
    train(step, model, optimizer,max_digit=train_length,max_digit_test=test_length)
    evaluate(model,max_test_digit=test_length)
    plt.subplot(211)
    global Loss_list
    global Accuracy_list_test
    plt.plot( Loss_list)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')

    plt.subplot(212)
    plt.plot( Accuracy_list_test)
    plt.xlabel('Iteration number / 10')
    plt.ylabel('Accuracy')
    plt.show()
   
   
