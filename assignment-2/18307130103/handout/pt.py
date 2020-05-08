import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 

from .data import prepare_batch, gen_data_batch, results_converter, prepare_batch_big


class myPTRNNModel(nn.Module):
    def __init__(self, choice = 'rnn'):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        if (choice == 'rnn'):
            self.rnn = nn.RNN(64, 64, 2, batch_first = True)
        if (choice == 'lstm'):
            self.rnn = nn.LSTM(64, 64, 2, batch_first = True)
        if (choice == 'gru'):
            self.rnn = nn.GRU(64, 64, 2, batch_first = True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        add1 = self.embed_layer(num1)
        add2 = self.embed_layer(num2)
        combined = torch.cat((add1, add2), 2) 
        rnnres, hidden = self.rnn(combined)
        logits = self.dense(rnnres)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.dense1 = nn.Linear(64, 64)
        self.rnn = nn.RNN(64, 64, 2, batch_first = True, nonlinearity = 'relu')
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        add1 = self.embed_layer(num1)
        add2 = self.embed_layer(num2)
        combined = torch.cat((add1, add2), 2) 
        combined = nn.functional.relu(self.dense1(combined))
#        combined = torch.tanh(self.dense1(combined))
        logits, hidden = self.rnn(combined)
        logits = self.dense(logits)
        return logits


class IRNNCell(nn.Module):
    """
    IndRNN Cell
    Performs a single time step operation
    """
    def __init__(self, inpdim, recdim, act=None):
        super().__init__()
        self.func = F.relu if act is None else act
        self.w = nn.Parameter(torch.randn(inpdim, recdim))
        self.u = nn.Parameter(torch.ones(recdim))
        self.b = nn.Parameter(torch.zeros(recdim))

    def forward(self, x_t, h_tm1):
        return self.func(F.linear(x_t, self.w, self.b) + torch.mul(h_tm1, self.u))

        
class IRNN(nn.Module):
    def __init__(self, input_size, output_size, depth=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        cells = []
        for i in range(depth):
            cells.append(IRNNCell(input_size, output_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        hidden = Variable(torch.zeros(x.size()[0], self.output_size))
        seq = []
        for i in range(x.size()[1]):
            x_t = x[:, i, :]
            for cell in self.cells:
                hidden = cell.forward(x_t, hidden)
            seq.append(hidden)
        return torch.stack(seq, dim=1)


class myAdvPTIRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.dense1 = nn.Linear(64, 64)
#        self.rnn = IndRNN(64, 64, batch_first=True)
        self.rnn = IRNN(64, 64, 2)
        self.dense = nn.Linear(64, 10)
        self.zero_grad()

    def forward(self, num1, num2):
        add1 = self.embed_layer(num1)
        add2 = self.embed_layer(num2)
        combined = torch.cat((add1, add2), 2) 
        combined = nn.functional.relu(self.dense1(combined))
        logits = self.rnn(combined)
#        logits, hidden = self.rnn(combined)
        logits = self.dense(logits)
        return logits


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    losses = losses
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


def train(epoch, batch_size, model, optimizer):
    loss = 0.0
    accuracy = 0.0
    for step in range(epoch):
        datas = gen_data_batch(batch_size=batch_size, start=0, end=500000000)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        if step % 50 == 0:
            print('step', step, ': loss', loss)    

def train_big(epoch, batch_size, model, optimizer):
    loss = 0.0
    accuracy = 0.0
    for step in range(epoch):
        Nums1, Nums2, results = prepare_batch_big(batch_size=batch_size, maxlen=51)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        if step % 50 == 0:
            print('step', step, ': loss', loss)


def evaluate_big(model, maxlen = 51):
    batch_size = 5000
    Nums1, Nums2, results = prepare_batch_big(batch_size, maxlen=maxlen)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    
    ok = 0
    for i in range(batch_size):
        if all(pred[i] == results[i]):
            ok = ok + 1
    acc = ok / batch_size
    print('accuracy is: %g' % acc)
    return(acc)

def evaluate(model):
    datas = gen_data_batch(batch_size=10000, start=0, end=1000000000)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])
    
    acc = np.mean([o[0]==o[1] for o in zip(datas[2], res)])
    print('accuracy is: %g' % acc)
    return(acc)


def pt_main(choice = 'rnn', epoch = 500, batch_size = 200, train_set = 'normal', evaluate_set = 'normal'):
    model = myPTRNNModel(choice = choice)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    if train_set == 'normal':
        train(epoch, batch_size, model, optimizer)
    else:
        train_big(epoch, batch_size, model, optimizer)
    if evaluate_set == 'normal':
        evaluate(model)
    else:
        evaluate_big(model)

def pt_adv_main(choice = 'rnn', epoch = 500, batch_size = 200, train_set = 'normal', evaluate_set = 'normal'):
    if choice == 'rnn':
        model = myAdvPTRNNModel()
    else:
        model = myAdvPTIRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    if train_set == 'normal':
        train(epoch, batch_size, model, optimizer)
    else:
        train_big(epoch, batch_size, model, optimizer)
    if evaluate_set == 'normal':
        evaluate(model)
    else:
        evaluate_big(model)