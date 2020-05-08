
import numpy as np
import torch
import torch.nn as nn

from .data import prepare_batch, gen_data_batch, results_converter
from .data import gen_data_batch2

class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2, batch_first=True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''

        a1 = self.embed_layer(num1)
        a2 = self.embed_layer(num2)
        logits = torch.cat((a1, a2), dim=2)
        logits = self.rnn(logits, None)[0]
        logits = self.dense(logits)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self, dim=32, hidden_size=64, num_layers=2, dropout=0, bi=False, model='RNN'):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.embed_layer = nn.Embedding(10, dim)
        if model=='RNN':
            self.rnn = nn.RNN(dim << 1, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bi)
        elif model=='LSTM':
            self.rnn = nn.LSTM(dim << 1, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bi)
        elif model=='GRU':
            self.rnn = nn.GRU(dim << 1, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bi)
        else:
            raise RuntimeError
        self.dense = nn.Linear(hidden_size, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        a1 = self.embed_layer(num1)
        a2 = self.embed_layer(num2)
        logits = torch.cat((a1, a2), dim=2)
        logits = self.rnn(logits, None)[0]
        logits = self.dense(logits)
        return logits


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


def train(steps, model, optimizer):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=0, end=555555555)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        if step % 50 == 0:
            print('step', step, ': loss', loss)

    return loss


def evaluate(model):
    datas = gen_data_batch(batch_size=2000, start=555555555, end=999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))


def pt_main():
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    # train(3000, model, optimizer)
    train(500, model, optimizer)
    evaluate(model)


def train2(steps, model, optimizer, length):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        Nums1, Nums2, results = gen_data_batch2(length, batch_size=200)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        if step % 50 == 0:
            print('step', step, ': loss', loss)

    return loss

def evaluate2(model, length):
    batch_size = 2000
    Nums1, Nums2, results = gen_data_batch2(length, batch_size)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = 0.0
    for i in range(batch_size):
        if all(pred[i] == results[i]):
            res += 1
    # print(type(res))
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % (res/batch_size))

def pt_adv_main(dim=128, hidden_size=64, num_layers=2, dropout=0, bi=False, model_name='RNN', steps=500, train_length=10, sequence_length=100):
    '''
    Please finish your code here.
    '''

    # gen_data_batch2(3, 25)
    model = myAdvPTRNNModel(dim, hidden_size, num_layers, dropout, bi, model_name)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train2(steps, model, optimizer, train_length)
    evaluate2(model, sequence_length)
    # pass
