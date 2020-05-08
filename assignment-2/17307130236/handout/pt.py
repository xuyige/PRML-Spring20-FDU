import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .data import prepare_batch, gen_data_batch, gen_great_data_batch, results_converter


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2, batch_first=True)
        self.dense = nn.Linear(64, 10)
        self.loss = []
        self.acc = []

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        # logits1,2=(batch_size,seq_len,input_size)
        logits1 = self.embed_layer(num1)
        logits2 = self.embed_layer(num2)
        input_logits = torch.cat([logits1, logits2], dim=2)
        # input_logits=(batch_size,seq_len,2*input_size)
        out, h_n = self.rnn(input_logits)
        logits = self.dense(out)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_layer = nn.Embedding(self.num_classes, self.input_size)
        self.rnn = nn.RNN(2*self.input_size, self.hidden_size,
                          self.num_layers, batch_first=True)
        self.dense = nn.Linear(self.hidden_size, self.num_classes)
        self.loss = []
        self.acc = []

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        h_state = None
        logits1 = self.embed_layer(num1)
        logits2 = self.embed_layer(num2)
        input_logits = torch.cat([logits1, logits2], dim=2)
        # input_logits=(batch_size,seq_len,2*input_size)
        out, h_n = self.rnn(input_logits)
        logits = self.dense(out)
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
    acc = 0.0
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=0, end=1e8)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=9)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        with torch.no_grad():
            logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
        logits = logits.numpy()
        pred = np.argmax(logits, axis=-1)
        res = results_converter(pred)
        acc = np.mean([o[0] == o[1] for o in zip(datas[2], res)])

        if step % 50 == 0:
            print('step', step, ': loss', loss)
            model.loss.append(loss)
            model.acc.append(acc)

    return loss


def my_train(steps, model, optimizer, batch_size, numdigit):
    loss = 0.0
    acc = 0.0
    for step in range(steps):
        datas = gen_great_data_batch(batch_size, numdigit)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=numdigit+2)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        with torch.no_grad():
            logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
        logits = logits.numpy()
        pred = np.argmax(logits, axis=-1)
        res = results_converter(pred)
        acc = np.mean([o[0] == o[1] for o in zip(datas[2], res)])

        if step % 50 == 0:
            print('step', step, ': loss', loss)
            model.loss.append(loss)
            model.acc.append(acc)

    return loss


def evaluate(model):
    datas = gen_data_batch(batch_size=3, start=0, end=1e8)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=9)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' %
          np.mean([o[0] == o[1] for o in zip(datas[2], res)]))


def my_evaluate(model, batch_size, numdigit):
    datas = gen_great_data_batch(batch_size, numdigit)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=numdigit+2)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' %
          np.mean([o[0] == o[1] for o in zip(datas[2], res)]))


def plot_metrics(model):
    """
    Visualize the trend of loss.
    """
    plt.plot(np.arange(len(model.loss)), model.loss)
    plt.title('Log-likelyhood')
    plt.show()


def draw_accuracy(model):
    """
    Visualize the trend of loss.
    """
    plt.plot(np.arange(len(model.acc)), model.acc)
    plt.title('Accuracy')
    plt.show()


def pt_main():
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(3000, model, optimizer)
    plot_metrics(model)
    draw_accuracy(model)
    evaluate(model)


def pt_adv_main():
    '''
    Please finish your code here.
    '''
    model = myAdvPTRNNModel(
        input_size=64, hidden_size=39, num_layers=2, num_classes=10)
    optimizer1 = torch.optim.Adam(params=model.parameters(), lr=0.01)
    my_train(3000, model, optimizer1, 200, 9)
    plot_metrics(model)
    draw_accuracy(model)
    my_evaluate(model, 2000, 9)
