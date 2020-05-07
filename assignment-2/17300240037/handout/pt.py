import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func


from .data import prepare_batch, gen_data_batch, results_converter, gen_data_batch_long


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, batch_first = True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = num1.cuda()
        num2 = num2.cuda()

        x = self.embed_layer(num1)
        y = self.embed_layer(num2)

        logits = torch.cat((x, y), dim = 2)

        logits, h = self.rnn(logits)

        logits = self.dense(logits)

        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.embed_layer = nn.Embedding(10, 64)
        self.rnn = nn.RNN(128, 128, batch_first = True)
        self.dense = nn.Linear(128, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = num1.cuda()
        num2 = num2.cuda()

        x = self.embed_layer(num1)
        y = self.embed_layer(num2)

        logits = torch.cat((x, y), dim = 2)

        logits, h = self.rnn(logits)

        logits = self.dense(logits)

        return logits

class myAdvPTRNNModel2(nn.Module):
    def __init__(self):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2, batch_first = True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = num1.cuda()
        num2 = num2.cuda()

        x = self.embed_layer(num1)
        y = self.embed_layer(num2)

        logits = torch.cat((x, y), dim = 2)

        logits, h = self.rnn(logits)

        logits = self.dense(logits)

        return logits
'''
class myAdvPTRNNModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.LSTM(64, 64, batch_first = True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        num1 = num1.cuda()
        num2 = num2.cuda()

        x = self.embed_layer(num1)
        y = self.embed_layer(num2)

        logits = torch.cat((x, y), dim = 2)

        logits, h = self.rnn(logits)

        logits = self.dense(logits)

        return logits
'''

def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.cuda().view(-1))


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
    logits = logits.cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))


def train_long(steps, model, optimizer):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        Nums1, Nums2, results = gen_data_batch_long(batch_size=200, length = 500)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        if step % 50 == 0:
            print('step', step, ': loss', loss)

    return loss

def evaluate_long(model):
    Nums1, Nums2, results = gen_data_batch_long(batch_size=2000, length = 500)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    results = results_converter(results)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(results, res)]))


def pt_main():
    model = myPTRNNModel().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer)
    evaluate(model)


def pt_adv_main():
    model = myAdvPTRNNModel().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train_long(500, model, optimizer)
    evaluate_long(model)

def pt_model1_main():
    model = myAdvPTRNNModel2().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train_long(500, model, optimizer)
    evaluate_long(model)
