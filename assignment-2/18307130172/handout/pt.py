
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt

from .data import prepare_batch, gen_data_batch, results_converter

import random


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 1)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1: torch.Tensor, num2: torch.Tensor):
        x1 = self.embed_layer(num1)
        x2 = self.embed_layer(num2)
        x = torch.cat((x1, x2), dim=2).transpose(0, 1)
        h = self.rnn(x)[0]
        y_pred = self.dense(h)
        return y_pred.transpose(0, 1).clone()


class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 21)
        self.fc1 = nn.Linear(21, 10)
        self.fc2 = nn.Linear(21, 21)
        # self.adder = nn.RNNCell(21, 21, nonlinearity="relu")
        self.adder = nn.GRUCell(21, 21)

    @staticmethod
    def generate_data(batch_size: int, start: int, end: int, transpose: bool = True, device = None):
        def random_seq():
            return [random.randrange(start, end) for i in range(batch_size)]

        def as_digits(seq, seq_len):
            seq0 = seq if type(seq[0]) == str else map(str, seq)
            return [
                list(map(int, s[::-1])) + [0] * (seq_len - len(s))
                for s in seq0
            ]

        x = random_seq()
        y = random_seq()
        z = [str(x0 + y0) for x0, y0 in zip(x, y)]
        seq_len = max(len(s) for s in z)

        if device is None:
            device = torch.device('cpu')
        x = torch.tensor(as_digits(x, seq_len), device=device)
        y = torch.tensor(as_digits(y, seq_len), device=device)
        z = torch.tensor(as_digits(z, seq_len), device=device)

        if transpose:
            return x.t(), y.t(), z.t()
        else:
            return x, y, z

    def forward(self, x0, y0, device=None):
        if device is None:
            device = torch.device('cpu')

        x = self.embed(x0)
        y = self.embed(y0)

        seq_len, batch_size = x.size()[:2]
        h = torch.zeros(batch_size, 21, device=device)
        z_pred = torch.empty(seq_len, batch_size, 10, device=device)
        for i in range(seq_len):
            g = self.adder(x[i], y[i])
            h0 = self.adder(g, h)
            z_pred[i] = self.fc1(h0)
            h = self.fc2(h0).relu()
            # h = h0

        return z_pred


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.view(-1))


def train_one_step(model, optimizer, x, y, label, dev):
    model.train()
    optimizer.zero_grad()
    logits = model(torch.tensor(x, device=dev), torch.tensor(y, device=dev))
    loss = compute_loss(logits, torch.tensor(label, device=dev))

    # compute gradient
    loss.backward()
    optimizer.step()
    return loss.item()


def train(steps, model, optimizer, num_digits, dev, mp):
    loss = 0.0
    accuracy = 0.0

    loss_history = []
    for step in range(steps):
        # datas = gen_data_batch(batch_size=64, start=0, end=5 * 10**num_digits)
        datas = gen_data_batch(batch_size=64, start=0, end=10**(num_digits + 1))

        Nums1, Nums2, results = prepare_batch(*datas, mp, maxlen=num_digits + 2)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results, dev)
        if (step + 1) % 10 == 0:
            # for o in list(zip(results, Nums1, Nums2))[:2]:
            #     print(o[1], o[2], o[0])
            loss_history.append(loss)
            print('step', step+1, ': loss', loss)

    return loss_history


def evaluate(model, num_digits, dev, mp):
    # datas = gen_data_batch(batch_size=10000, start=5 * 10**num_digits, end=10**(num_digits + 1))
    datas = gen_data_batch(batch_size=10000, start=0, end=10**(num_digits + 1))

    Nums1, Nums2, results = prepare_batch(*datas, mp, maxlen=num_digits + 2)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1, device=dev), torch.tensor(Nums2, device=dev))
    logits = logits.cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    real_ans = results_converter(results)

    # Test 1 + 999...999
    # with torch.no_grad():
    #     x0 = torch.tensor([[1] + [0] * num_digits], device=dev)
    #     y0 = torch.tensor([[9] * num_digits + [0]], device=dev)
    #     z0_pred = model(x0, y0).argmax(-1)
    #     print(x0)
    #     print(y0)
    #     print(z0_pred)

    # for o in list(zip(real_ans, res, Nums1, Nums2))[:20]:
    #     print(o[2], o[3], o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(real_ans, res)]))


def require_device(use_cuda):
    if use_cuda and not torch.cuda.is_available():
        print("(warn) Cuda not available!")

    if use_cuda and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    return dev


def pt_main(num_digits, mp, use_cuda=False):
    dev = require_device(use_cuda)
    model = myPTRNNModel().to(dev)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.03)
    loss_history = train(500, model, optimizer, num_digits, dev, mp)
    evaluate(model, num_digits, dev, mp)

    return loss_history


def pt_adv_main(num_digits, use_cuda=False):
    dev = require_device(use_cuda)
    model = myAdvPTRNNModel().to(dev)
    optimizer = opt.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    NUM_STEPS = 500
    BATCH_SIZE = 64
    TEST_SIZE = 10000
    CHECKOUT_INTERVAL = 10

    loss_history = []
    for i in range(NUM_STEPS):
        x, y, z = model.generate_data(BATCH_SIZE, 0, 5 * 10**num_digits, device=dev)
        x, y, z = model.generate_data(BATCH_SIZE, 0, 10**(num_digits + 1), device=dev)

        optimizer.zero_grad()
        z_pred = model(x, y, device=dev)
        loss = criterion(z_pred.reshape(-1, 10), z.reshape(-1))
        loss.backward()
        optimizer.step()

        if (i + 1) % CHECKOUT_INTERVAL == 0:
            loss_history.append(loss.item())
            print(f'[{i + 1}] loss = {loss.item()}')

    x, y, z = model.generate_data(
        TEST_SIZE,
        # 5 * 10**num_digits, 10**(num_digits + 1),
        0, 10**(num_digits + 1),
        transpose=False, device=dev
    )

    with torch.no_grad():
        z_pred = model(x.t(), y.t(), device=dev).transpose(0, 1).argmax(-1)

        # Test 1 + 999...999
        # x0 = torch.tensor([[1] + [0] * num_digits], device=dev)
        # y0 = torch.tensor([[9] * num_digits + [0]], device=dev)
        # z0_pred = model(x0.t(), y0.t(), device=dev).transpose(0, 1).argmax(-1)
        # print(x0)
        # print(y0)
        # print(z0_pred)

    correct_cnt = (z == z_pred).all(-1).sum().item()
    print(f'Accuracy: {correct_cnt / TEST_SIZE}')

    return loss_history
