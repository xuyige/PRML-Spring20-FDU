
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json

from .data import prepare_batch, gen_data_batch, results_converter


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = []
        self.acc = []
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2, batch_first=True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        n1 = self.embed_layer(num1)
        n2 = self.embed_layer(num2)
        input = torch.cat([n1,n2],-1)
        output, hidden = self.rnn(input,None)
        logits = self.dense(output)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = []
        self.acc = []
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2, batch_first=True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        n1 = self.embed_layer(num1)
        n2 = self.embed_layer(num2)
        input = torch.cat([n1,n2],-1)
        output, hidden = self.rnn(input,None)
        logits = self.dense(output)
        return logits

def show(model):
    plt.plot(model.loss)
    plt.xlabel('batches')
    plt.ylabel('Loss')
    plt.show()
    

def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.view(-1))


def train_one_step(model, optimizer, x, y, label):
    model.train()
    optimizer.zero_grad()
    logits = model(torch.tensor(x), torch.tensor(y))
    loss = compute_loss(logits, torch.tensor(label))

    # compute gradient
    loss.backward()
    optimizer.step()
    return loss.item()


def train(steps, model, optimizer, length):
    loss = 0.0
    accuracy = 0.0
    start = 0
    end = 0
    for i in range(length): end = end*10+5
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=start, end=end)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=length+2)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        model.loss.append(loss)   
        if step % 50 == 0:
            print('step', step, ': loss', loss)

    return loss


def evaluate(model, length, output):
    start = 0
    end = 0
    for i in range(length): start = start*10+5
    for i in range(length): end = end*10+9
    datas = gen_data_batch(batch_size=2000, start=start, end=end)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=length+2)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    #for o in list(zip(datas[2], res))[:20]:
    #    print(o[0], o[1], o[0]==o[1])
    acc = np.mean([o[0]==o[1] for o in zip(datas[2], res)])
    if output==1:
        print('accuracy is: %g' % acc)
    return acc

    
def pt_main():
    length = 10
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(1000, model, optimizer, length)
    evaluate(model, length, 1)
    show(model)


def pt_adv_main():
    length = 10
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.02)
    train(1000, model, optimizer, length)
    evaluate(model, length, 1)
    show(model)
