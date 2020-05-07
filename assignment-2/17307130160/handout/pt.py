import numpy as np
import torch
import torch.nn as nn
from .data import prepare_batch, gen_data_batch, results_converter
from matplotlib import pyplot as plt


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = num1.cuda()
        num2 = num2.cuda()

        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)

        In = torch.cat([num1, num2], 2).transpose(0, 1)
        Out, Hid = self.rnn(In)

        logits = self.dense(Out).transpose(0, 1).clone().contiguous()
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 3)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = num1.cuda()
        num2 = num2.cuda()

        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)

        In = torch.cat([num1, num2], 2).transpose(0, 1)
        Out, Hid = self.rnn(In)

        logits = self.dense(Out).transpose(0, 1).clone().contiguous()
        return logits


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
    end = 5*10**99
    for step in range(steps):
        datas = gen_data_batch(200, 0, end)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=101)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        st.append(step)
        lossed.append(loss)

        if step % 50 == 0:
            print('step', step, ': loss', loss)

    return loss


def evaluate(model):
    start = 5*10**99
    end = 10**100 - 1
    datas = gen_data_batch(3000, start, end)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=101)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.cuda().data.cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0] == o[1] for o in zip(datas[2], res)]))


st = []
lossed = []


def pt_main():
    st.clear()
    lossed.clear()
    model = myPTRNNModel().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(1000, model, optimizer)
    evaluate(model)
    x = np.array(st)
    y = np.array(lossed)
    plt.plot(x, y, color ="red")



def pt_adv_main():
    '''
    Please finish your code here.
    '''
    st.clear()
    lossed.clear()
    model = myAdvPTRNNModel().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    train(1000, model, optimizer)
    evaluate(model)
    x = np.array(st)
    y = np.array(lossed)
    plt.plot(x, y)
    plt.show()
