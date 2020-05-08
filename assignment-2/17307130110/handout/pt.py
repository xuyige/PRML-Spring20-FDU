
import numpy as np
import torch
import torch.nn as nn

from .data import prepare_batch, gen_data_batch, results_converter


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
        rnn_input_1 = self.embed_layer(num1.t())
        rnn_input_2 = self.embed_layer(num2.t())
        rnn_input = torch.cat((rnn_input_1, rnn_input_2), 2)
        output, hn = self.rnn(rnn_input)
        logits = self.dense(output).permute(1,0,2)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 1)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        rnn_input_1 = self.embed_layer(num1.t())
        rnn_input_2 = self.embed_layer(num2.t())
        rnn_input = torch.cat((rnn_input_1, rnn_input_2), 2)
        output, hn = self.rnn(rnn_input)
        logits = self.dense(output).permute(1,0,2)
        return logits


def compute_loss(logits, labels):
    # print("logits is ", logits.size())
    # print("labels is ", labels.size())
    losses = nn.CrossEntropyLoss()
    return losses(logits.contiguous().view(-1, 10), labels.view(-1))


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
    datas = gen_data_batch(batch_size=20000, start=555555555, end=999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %f' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))


def pt_main():
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(3000, model, optimizer)
    evaluate(model)


def pt_adv_main():
    '''
    Please finish your code here.
    '''
    model = myAdvPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    train(1000, model, optimizer)
    evaluate(model)
