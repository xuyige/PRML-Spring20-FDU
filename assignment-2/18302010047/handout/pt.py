
import numpy as np
import torch
import torch.nn as nn
# import fitlog


from .data import prepare_batch, gen_data_batch, results_converter


class myPTRNNModel(nn.Module):
    def __init__(self, dim=32, layers=2):
        super().__init__()
        self.embed_layer = nn.Embedding(10, dim)
        self.rnn = nn.RNN(dim*2, dim*2, layers, batch_first=True)
        self.dense = nn.Linear(dim*2, 10)

    def forward(self, num1, num2):
        x1 = self.embed_layer(num1)
        x2 = self.embed_layer(num2)
        x = torch.cat((x1, x2), -1)
        logits, h = self.rnn(x)
        logits = self.dense(logits)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self, dim=32, layers=2, model="LSTM"):
        super().__init__()
        self.embed_layer = nn.Embedding(10, dim)
        if model == "LSTM":
            self.rnn = nn.LSTM(dim*2, dim*2, layers, batch_first=True)
        elif model == "GRU":
            self.rnn = nn.GRU(dim*2, dim*2, layers, batch_first=True)
        elif model == "Transformers":
            pass
        self.dense = nn.Linear(dim*2, 10)
        

    def forward(self, num1, num2):
        x1 = self.embed_layer(num1)
        x2 = self.embed_layer(num2)
        x = torch.cat((x1, x2), -1)
        logits, h = self.rnn(x)
        logits = self.dense(logits)
        return logits


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.view(-1))


def train_one_step(model, optimizer, x, y, label, truth):
    model.train()
    optimizer.zero_grad()
    logits = model(torch.tensor(x), torch.tensor(y))
    loss = compute_loss(logits, torch.tensor(label))
    # compute gradient
    loss.backward()
    optimizer.step()

    logits = logits.detach().numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    acc = np.mean([o[0]==o[1] for o in zip(truth, res)])

    return loss.item(), acc


def train(steps, model, optimizer):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=0, end=10 ** 10)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
        loss, accuracy = train_one_step(model, optimizer, Nums1, Nums2, results, datas[2])
        # fitlog.add_loss(loss, name="Loss", step=step)
        # fitlog.add_metric(accuracy, name="Accuracy", step=step)
        if step % 50 == 0:
            print('step', step, ': loss', loss, ' : acc', accuracy)

    return loss


def evaluate(model):
    datas = gen_data_batch(batch_size=2000, start=0, end=10 ** 10)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    for o in list(zip(datas[2], res))[:20]:
        print(o[0], o[1], o[0] == o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))


def pt_main(dim, layers, epochs, lr):
    model = myPTRNNModel(dim=dim, layers=layers)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train(epochs, model, optimizer)
    evaluate(model)


def pt_adv_main(dim, layers, epochs, lr, model):
    model = myAdvPTRNNModel(dim, layers, model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train(epochs, model, optimizer)
    evaluate(model)
