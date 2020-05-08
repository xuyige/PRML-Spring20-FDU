
import numpy as np
import torch
import torch.nn as nn

from .data import prepare_batch, gen_data_batch, results_converter, gen_big_data_batch


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 10)
        self.rnn = nn.RNN(20, 20, batch_first = True)
        self.dense = nn.Linear(20, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = num1.cuda()
        num2 = num2.cuda()

        x1 = self.embed_layer(num1)
        x2 = self.embed_layer(num2)

        input = torch.cat((x1, x2), dim = 2)

        output, h = self.rnn(input)

        logits = self.dense(output)
        
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self):
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

        x1 = self.embed_layer(num1)
        x2 = self.embed_layer(num2)

        input = torch.cat((x1, x2), dim = 2)

        output, h = self.rnn(input)

        logits = self.dense(output)
        
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
    for step in range(steps):
        Nums1, Nums2, results = gen_big_data_batch(batch_size=200, length = 100)
        #Nums1, Nums2, results = gen_data_batch_long(batch_size=10, length = 10)
        #print(len(Nums1), len(Nums2), len(results))
        #print(Nums1)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        if step % 50 == 0:
            print('step', step, ': loss', loss)

    return loss


def evaluate(model):
    Nums1, Nums2, results = gen_big_data_batch(batch_size=2000, length = 100)
    #Nums1, Nums2, results = gen_data_batch_long(batch_size=1000, length = 100)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    results = results_converter(results)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(results, res)]))



def pt_main(steps):
    model = myPTRNNModel().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(steps, model, optimizer)
    evaluate(model)


def pt_adv_main(steps):
    model = myAdvPTRNNModel().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(steps, model, optimizer)
    evaluate(model)
    pass

