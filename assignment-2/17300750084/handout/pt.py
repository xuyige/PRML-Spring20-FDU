import numpy as np
import torch
import torch.nn as nn
import argparse
# from torch.utils.tensorboard import SummaryWriter
from handout.data import prepare_batch, gen_data_batch, results_converter

# writer = SummaryWriter('./data/')
n_layers= 3
n_digits = 100
n_iters = 1000
type = 'rnn'


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32) # embedding 层
        self.rnn = nn.RNN(64, 64, 2) # rnn层：输入为64bit 隐含层有64个神经元 有2层隐藏层
        self.dense = nn.Linear(64, 10) #

    def forward(self, num1, num2):
        x1 = self.embed_layer(num1)
        x2 = self.embed_layer(num2)
        input = torch.cat((x1, x2), dim=2, out=None).transpose(0, 1)
        output, hidden = self.rnn(input)
        logits = self.dense(output)
        logits = logits.transpose(0, 1)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self, n_layer, type):
        self.hidden_layer = n_layer
        self.hidden_dim = 64
        super().__init__()
        if type =='rnn':
            self.init_rnn()
        elif type =='lstm':
            self.init_lstm()
        elif type =='gru':
            self.init_gru()


    def init_gru(self):
        self.type='gru'
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=self.hidden_dim,
            num_layers=self.hidden_layer,
            dropout=0
        )
        self.embed_layer = nn.Embedding(10, 32)
        self.dense = nn.Linear(64,10)

    def gru_forward(self, num1, num2, h):
        x1 = self.embed_layer(num1)
        x2 = self.embed_layer(num2)
        input = torch.cat((x1, x2), dim=2, out=None).transpose(0, 1)
        output, h = self.gru(input, h)
        logits = self.dense(output)
        logits = logits.transpose(0, 1)
        # self.hidden = self.hidden.detach()
        return logits, h

    def init_lstm(self):
        self.type = 'lstm'
        self.embed_layer = nn.Embedding(10, 32)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=self.hidden_dim,
            num_layers =self.hidden_layer,
            dropout=0)
        self.dense = nn.Linear(64, 10)


    def lstm_forward(self, num1, num2, h, c):
        x1 = self.embed_layer(num1)
        x2 = self.embed_layer(num2)
        input = torch.cat((x1, x2), dim=2, out=None).transpose(0, 1)
        output, (h, c) = self.lstm(input, (h, c))
        output = self.dense(output)
        output = output.transpose(0, 1)
        return output, h, c

    def init_hidden(self, input_size):
        return torch.zeros(self.hidden_layer, input_size, self.hidden_dim)

    def init_cell(self, input_size):
        return torch.zeros(self.hidden_layer, input_size, self.hidden_dim)

    def init_rnn(self):
        self.type = 'rnn'
        self.embed_layer = nn.Embedding(10, 32)  # embedding 层
        self.rnn = nn.RNN(
            input_size=64,
            hidden_size=self.hidden_dim,
            num_layers=self.hidden_layer)  # rnn层：输入为64bit 隐含层有64个神经元 有2层隐藏层

        self.dense = nn.Linear(64, 10)  #

    def rnn_forward(self, num1, num2, h):
        x1 = self.embed_layer(num1)
        x2 = self.embed_layer(num2)
        input = torch.cat((x1, x2), dim=2, out=None).transpose(0, 1)
        output, h = self.rnn(input, h)
        logits = self.dense(output)
        # print(output.shape, logits.shape)
        logits = logits.transpose(0, 1)
        return logits, h

    def forward(self, num1, num2, h=None, c=None):
        # print(self.type)
        if self.type == 'rnn':
            return self.rnn_forward(num1, num2, h)
        elif self.type =='lstm':
            return self.lstm_forward(num1, num2, h, c)
        elif self.type =='gru':
            return self.gru_forward(num1, num2, h)





def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    # x = logits.reshape(-1, 10)
    # y = labels.view(-1)
    # print(labels.shape, x.shape, y.shape)
    return losses(logits.reshape(-1, 10), labels.view(-1))


def train_one_step(model, optimizer, x, y, label):
    model.train()
    optimizer.zero_grad()
    logits = model(torch.tensor(x), torch.tensor(y))
    loss = compute_loss(logits, torch.tensor(label))
    # compute gradient
    loss.backward()
    optimizer.step()
    return loss.item()


def train_one_step_bits(model, optimizer, x, y, label, batch_size, type):
    model.train()
    optimizer.zero_grad()
    if type == 'rnn' or type =='gru':
        h = model.init_hidden(batch_size)
        logits, h = model(torch.tensor(x), torch.tensor(y), h)
    else:
        h = model.init_hidden(batch_size)
        c = model.init_cell(batch_size)
        logits, h, c = model(torch.tensor(x), torch.tensor(y), h, c)
        # print(logits.shape, h.shape, c.shape)

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
        loss = train_one_step(model, optimizer, Nums1,Nums2, results)
        # writer.add_scalar('[%s] loss with %d digits / %d hidden layers' % ('rnn', 11, 2), loss, step)

        if step % 50 == 0:
            print('step', step, ': loss', loss)

    return loss

def train_bits(steps, model, optimizer, scheduler, n_maxlen, n_layer, type):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=0, end=555555555)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=n_maxlen)
        loss = train_one_step_bits(model, optimizer, Nums1,Nums2, results, 200, type)
        # writer.add_scalar('[%s] loss with %d digits / %d hidden layers' % (type, n_maxlen, n_layer), loss, step)

        if loss < 1e-3:
            return loss
        scheduler.step(loss)
        if step % 50 == 0:
            print('step', step, ': loss', loss)

    return loss

def evaluate_bits(model, n_maxlen, type):
    datas = gen_data_batch(batch_size=2000, start=555555555, end=999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=n_maxlen)
    with torch.no_grad():
        if type == 'rnn' or type =='gru':
            h = model.init_hidden(2000)
            logits, h = model(torch.tensor(Nums1), torch.tensor(Nums2), h)
        else:
            h = model.init_hidden(2000)
            c = model.init_cell(2000)
            logits, h,c = model(torch.tensor(Nums1), torch.tensor(Nums2), h,c)
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    print('accuracy is: %g' % np.mean([o[0] == o[1] for o in zip(datas[2], res)]))

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
    train(3000, model, optimizer)
    evaluate(model)



def pt_adv_main(arg):
    n_layers = arg.layers
    n_iters = arg.iters
    type = arg.type
    n_digits = arg.len

    model = myAdvPTRNNModel(n_layers, type)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                               mode='min',
                                               factor=0.5,
                                               patience=10,
                                               verbose=True,
                                               threshold=0.001,
                                               threshold_mode='rel',
                                               min_lr=0.001,
                                               )
    print('%s model with %d hidden layers \n input digit-length: %d \n training iterations: %d ' %(type, n_layers, n_digits, n_iters))
    train_bits(n_iters, model, optimizer, scheduler, n_digits, n_layers, type)
    evaluate_bits(model, n_digits, type)