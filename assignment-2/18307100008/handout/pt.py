import numpy as np
import torch
import torch.nn as nn

from .data import prepare_batch, gen_data_batch, results_converter

MAX_LENGTH = 9


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        embed1 = self.embed_layer(num1)
        embed2 = self.embed_layer(num2)
        input = torch.cat((embed1, embed2), 2)
        h0 = torch.randn(2, input.shape[1], 64)
        output = self.rnn(input, h0)[0]
        logits = self.dense(output)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.encoder = nn.GRU(64, 64, 1)
        self.decoder = nn.GRU(64, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, num1, num2):
        embed1 = self.embed_layer(num1)
        embed2 = self.embed_layer(num2)
        input = torch.cat((embed1, embed2), 2)
        rnn_output, hidden = self.encoder(input)
        output = self.decoder(rnn_output, hidden)
        logits = self.out(output)
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


def train(steps, model, optimizer, lr_reduction=False, length=MAX_LENGTH):
    loss = 0.0
    accuracy = 0.0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=int('5' * length), end=int('9' * length))
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=length+1)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        if lr_reduction:
            scheduler.step()
        if step % 50 == 0:
            print('step', step, ': loss', loss)

    return loss


def evaluate(model, length=MAX_LENGTH):
    datas = gen_data_batch(batch_size=2000, start=int('5' * length), end=int('9' * length))
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=length+1)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    print(logits)
    pred = np.argmax(logits, axis=-1)
    print(pred[:100])
    res = results_converter(pred)
    for o in list(zip(datas[2], res))[:20]:
        print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0] == o[1] for o in zip(datas[2], res)]))


def pt_main():
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(3000, model, optimizer)
    evaluate(model)


def pt_adv_main():
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(3000, model, optimizer)
    evaluate(model)
