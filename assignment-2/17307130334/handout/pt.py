
import numpy as np
import torch
import torch.nn as nn
import time
from .data import prepare_batch, gen_data_batch, results_converter

length=50

class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(length, 32)
        self.rnn = nn.RNN(64, 64, 1, batch_first=True) # input_size, hidden_size, num_layers
        self.dense = nn.Linear(64, length)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1=self.embed_layer(num1)
        num2=self.embed_layer(num2)
        input = torch.cat((num1, num2), 2)
        output,h_n=self.rnn(input)
        logits=self.dense(output)
        return logits

class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.embed_layer = nn.Embedding(length, 32)
        self.rnn = nn.RNN(64, 64, 2, batch_first=True) # input_size, hidden_size, num_layers
        # self.lstm = nn.LSTM(64, 64, batch_first=True)
        # self.gru = nn.GRU(64, 64, batch_first=True)
        self.dense = nn.Linear(64, length)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)

        input = torch.cat((num1, num2), 2)
        output, h_n = self.rnn(input)
        # output, h_n = self.lstm(input)
        # output, h_n = self.gru(input)

        logits = self.dense(output)

        return logits


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, length), labels.view(-1))


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

    # start = time.time()

    loss = 0.0
    loss_values=[]
    accuracy = 0.0
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=0, end=int(str((length-1)*'5')))
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=length+1) #11
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        loss_values.append(loss)

        if step % 50 == 0:
            print('step', step, ': loss', loss)

    # end= time.time()
    # time_taken=end-start

    return loss


def evaluate(model):
    datas = gen_data_batch(batch_size=2000, start=int(str((length-1)*'5')), end=int(str((length-1)*'9')))
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=length+1) #11
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)

    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))


round=1000
def pt_main():
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(round, model, optimizer)  # step:3000
    evaluate(model)


def pt_adv_main():
    '''
            Please finish your code here.
            '''
    model = myAdvPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    train(round, model, optimizer) #step:3000
    evaluate(model)



