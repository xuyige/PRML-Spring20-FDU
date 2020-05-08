
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from handout.data import *


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = []
        self.accuracy = []
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        x = self.embed_layer(num1)
        y = self.embed_layer(num2)
        sum = torch.cat((x, y), -1).transpose(0, 1)
        #print(num1.size())
        sum, h_state = self.rnn(sum)
        logits = self.dense(sum).transpose(0, 1).contiguous()
        return logits


#embed_num embeding层维度 rnn_num rnn层数 type rnn网络类型
class myAdvPTRNNModel(nn.Module):
    def __init__(self, embed_num, rnn_num, type):
        super().__init__()
        self.loss = []
        self.accuracy = []
        self.embed_layer = nn.Embedding(10, embed_num)
        if type == 'RNN':
            self.rnn = nn.RNN(2*embed_num, 2*embed_num, rnn_num)
        elif type == 'GRU':
            self.rnn = nn.GRU(2 * embed_num, 2 * embed_num, rnn_num)
        elif type == 'LSTM':
            self.rnn = nn.LSTM(2 * embed_num, 2 * embed_num, rnn_num)
        self.dense = nn.Linear(2*embed_num, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        x = self.embed_layer(num1)
        y = self.embed_layer(num2)
        sum = torch.cat((x, y), -1).transpose(0, 1)
        sum, h_state = self.rnn(sum)
        logits = self.dense(sum).transpose(0, 1).contiguous()
        return logits


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.view(-1))


#新增参数device用于标识使用cpu还是gpu
def train_one_step(model, optimizer, x, y, label, device):
    model.train()
    optimizer.zero_grad()
    logits = model(torch.tensor(x).to(device), torch.tensor(y).to(device))
    loss = compute_loss(logits, torch.tensor(label).to(device))

    # compute gradient
    loss.backward()
    optimizer.step()
    return loss.item()


#新增参数device用于标识用cpu还是gpu，digit_length标识训练集数据位数，evaluate_digit_length标识测试集数据位数
def train(steps, model, optimizer, device, digit_length, evaluate_digit_length):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        datas = gen_data_batch_large(batch_size=200, start=0, end=5*10**(digit_length-1))
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=digit_length+1)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results, device)
        model.loss.append(loss)
        if step % 10 == 0:
            accuracy = evaluate(model, device, evaluate_digit_length)
            model.accuracy.append(accuracy)
        if step % 50 == 0:
            print('step', step, ': loss', loss, ' accuracy', accuracy)

    return loss


def evaluate(model, device, digit_length):
    datas = gen_data_batch_large(batch_size=2000, start=5*10**(digit_length-1), end=1*10**digit_length-1)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=digit_length+1)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1).to(device),
                       torch.tensor(Nums2).to(device))
        logits = logits.cpu()
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])
    accuracy = np.mean([o[0] == o[1] for o in zip(datas[2], res)])
    return accuracy


def pt_main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = myPTRNNModel().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    plt.subplot(211)
    plt.plot(model.loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.subplot(212)
    accuracy_x = range(len(model.accuracy))
    accuracy_x = [10 * x for x in accuracy_x]
    plt.plot(accuracy_x, model.accuracy)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()
    return


#试验优化后的模型，没有包括比较试验的函数
def pt_adv_main():
    '''
    Please finish your code here.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = myAdvPTRNNModel(64, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    train(500, model, optimizer, device, 50, 50)
    loss_adv = model.loss
    accuracy_adv = model.accuracy
    model = myPTRNNModel().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss = model.loss
    accuracy = model.accuracy
    plt.subplot(211)
    plt.plot(loss, label='common')
    plt.plot(loss_adv, label='adv')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.subplot(212)
    accuracy_x = range(len(accuracy))
    accuracy_x = [10 * x for x in accuracy_x]
    plt.plot(accuracy_x, accuracy, label='common')
    plt.plot(accuracy_x, accuracy_adv, label='adv')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return


#embeding层维度比较试验
def embed_compare():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = myAdvPTRNNModel(16, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_16 = model.loss
    accuracy_16 = model.accuracy
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_32 = model.loss
    accuracy_32 = model.accuracy
    model = myAdvPTRNNModel(128, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_64 = model.loss
    accuracy_64 = model.accuracy
    plt.subplot(211)
    plt.plot(loss_16, label='embedding dimension: 16')
    plt.plot(loss_32, label='embedding dimension: 32')
    plt.plot(loss_64, label='embedding dimension: 64')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.subplot(212)
    accuracy_x = range(len(accuracy_16))
    accuracy_x = [10 * x for x in accuracy_x]
    plt.plot(accuracy_x, accuracy_16, label='embedding dimension: 16')
    plt.plot(accuracy_x, accuracy_32, label='embedding dimension: 32')
    plt.plot(accuracy_x, accuracy_64, label='embedding dimension: 128')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


#rrn层数比较试验
def rnn_layer_compare():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = myAdvPTRNNModel(32, 1, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(3000, model, optimizer, device, 3, 50)
    loss_1 = model.loss
    accuracy_1 = model.accuracy
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_2 = model.loss
    accuracy_2 = model.accuracy
    model = myAdvPTRNNModel(32, 3, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_3 = model.loss
    accuracy_3 = model.accuracy
    model = myAdvPTRNNModel(32, 4, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_4 = model.loss
    accuracy_4 = model.accuracy
    model = myAdvPTRNNModel(32, 5, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_5 = model.loss
    accuracy_5 = model.accuracy
    plt.subplot(211)
    plt.plot(loss_1, label='rnn layers: 1')
    plt.plot(loss_2, label='rnn layers: 2')
    plt.plot(loss_3, label='rnn_layers: 3')
    plt.plot(loss_4, label='rnn_layers: 4')
    plt.plot(loss_5, label='rnn_layers: 5')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.subplot(212)
    accuracy_x = range(len(accuracy_1))
    accuracy_x = [10 * x for x in accuracy_x]
    plt.plot(accuracy_x, accuracy_1, label='rnn layers: 1')
    plt.plot(accuracy_x, accuracy_2, label='rnn layers: 2')
    plt.plot(accuracy_x, accuracy_3, label='rnn layers: 3')
    plt.plot(accuracy_x, accuracy_4, label='rnn layers: 4')
    plt.plot(accuracy_x, accuracy_5, label='rnn layers: 5')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


#学习率比较试验
def lr_compare():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_0001 = model.loss
    accuracy_0001 = model.accuracy
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    train(500, model, optimizer, device, 50, 50)
    loss_001 = model.loss
    accuracy_001 = model.accuracy
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
    train(500, model, optimizer, device, 50, 50)
    loss_01 = model.loss
    accuracy_01 = model.accuracy
    plt.subplot(211)
    plt.plot(loss_0001, label='learning rate: 0.001')
    plt.plot(loss_001, label='learning rate: 0.01')
    plt.plot(loss_01, label='learning: 0.1')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.subplot(212)
    accuracy_x = range(len(accuracy_01))
    accuracy_x = [10 * x for x in accuracy_x]
    plt.plot(accuracy_x, accuracy_0001, label='learning rate: 0.001')
    plt.plot(accuracy_x, accuracy_001, label='learning rate: 0.01')
    plt.plot(accuracy_x, accuracy_01, label='learning rate: 0.1')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


#rnn网络类型比较试验
def rnn_type_compare():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_RNN = model.loss
    accuracy_RNN = model.accuracy
    model = myAdvPTRNNModel(32, 2, 'GRU').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_GRU = model.loss
    accuracy_GRU = model.accuracy
    model = myAdvPTRNNModel(32, 2, 'LSTM').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(500, model, optimizer, device, 50, 50)
    loss_LSTM = model.loss
    accuracy_LSTM = model.accuracy
    plt.subplot(211)
    plt.plot(loss_RNN, label='RNN')
    plt.plot(loss_GRU, label='GRU')
    plt.plot(loss_LSTM, label='LSTM')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.subplot(212)
    accuracy_x = range(len(accuracy_RNN))
    accuracy_x = [10 * x for x in accuracy_x]
    plt.plot(accuracy_x, accuracy_RNN, label='RNN')
    plt.plot(accuracy_x, accuracy_GRU, label='GRU')
    plt.plot(accuracy_x, accuracy_LSTM, label='LSTM')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


#泛化能力测试
def test_general_ability():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(1500, model, optimizer, device, 2, 50)
    accuracy_1 = model.accuracy
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(1500, model, optimizer, device, 3, 50)
    accuracy_2 = model.accuracy
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(1500, model, optimizer, device, 5, 50)
    accuracy_3 = model.accuracy
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(1500, model, optimizer, device, 10, 50)
    accuracy_4 = model.accuracy
    model = myAdvPTRNNModel(32, 2, 'RNN').to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(1500, model, optimizer, device, 20, 50)
    accuracy_5 = model.accuracy
    accuracy_x = range(len(accuracy_1))
    accuracy_x = [10 * x for x in accuracy_x]
    plt.plot(accuracy_x, accuracy_1, label='train digits length: 2')
    plt.plot(accuracy_x, accuracy_2, label='train digits length: 3')
    plt.plot(accuracy_x, accuracy_3, label='train digits length: 5')
    plt.plot(accuracy_x, accuracy_4, label='train digits length: 10')
    plt.plot(accuracy_x, accuracy_5, label='train digits length: 20')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
