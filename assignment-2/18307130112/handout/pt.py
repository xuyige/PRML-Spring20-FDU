
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from numpy.random import uniform
from .data import *



class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2, batch_first=True)
        self.dense = nn.Linear(64, 10)
        self.loss, self.accuracy = [], []        

    def forward(self, num1, num2):
        number1 = self.embed_layer(num1)
        number2 = self.embed_layer(num2)
        number = torch.cat((number1, number2),2)
        logits, non = self.rnn(number)
        logits = self.dense(logits)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self, dim = 32, layers = 2, mod = "GRU"):
        super().__init__()
        self.embed_layer = nn.Embedding(10, dim)
        if mod=="GRU":
            self.rnn = nn.GRU(dim * 2, dim * 2, layers, batch_first=True)
        elif mod=="LSTM":
            self.rnn = nn.LSTM(dim * 2, dim * 2, layers, batch_first=True)
        else:
            self.rnn = nn.RNN(dim * 2, dim * 2, layers, batch_first=True)
        self.dense = nn.Linear(dim * 2, 10)
        self.loss, self.accuracy = [], [] 

    def forward(self, num1, num2):
        number1 = self.embed_layer(num1)
        number2 = self.embed_layer(num2)
        number = torch.cat((number1, number2), -1)
        logits, non = self.rnn(number)
        logits = self.dense(logits)
        return logits

def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.view(-1))

def train_one_step(model, optimizer, x, y, label, device = "cpu"):
    model.train()
    optimizer.zero_grad()
    logits = model(torch.LongTensor(x).to(device), torch.LongTensor(y).to(device))
    loss = compute_loss(logits, torch.LongTensor(label).to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


def train(steps, model, optimizer, device = "cpu", diglength = 10):
    loss = 0.0
    end =5*10**(diglength-1)
    for step in range(steps):
        #print(step)
        datas = gen_data_batch(batch_size=200, start=0, end=end)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=diglength+1)
        loss = train_one_step(model, optimizer, Nums1, Nums2, results, device)
        model.loss.append(loss)
        if step % 10 == 0:
            model.accuracy.append(evaluate(model,device,diglength))
        #model.accuracy.append(evaluate(model,device,diglength))
        if step % 50 == 0:
            print("step", step, ": loss", loss)
    return loss


def evaluate(model, device = "cpu", diglength = 10):
    start = 5*10**(diglength - 1)+1
    end = 10**diglength - 1
    datas = gen_data_batch(batch_size=2000, start=start, end=end)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=diglength+1)
    with torch.no_grad():
        logits = model(torch.LongTensor(Nums1).to(device), torch.LongTensor(Nums2).to(device))
        logits = logits.cpu()
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    return np.mean([o[0]==o[1] for o in zip(datas[2], res)])

def pt_main():
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(1000, model, optimizer, "cpu", 10)
    print("Accuracy = ", evaluate(model, "cpu", 10))
    plt.subplot(2, 1, 1)
    StepLos = np.arange(stop = step)
    plt.plot(StepLos, model.loss)
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(2, 1, 2)
    StepAcc = np.arange(stop = step, step = 10)
    plt.plot(StepAcc, model.accuracy)
    plt.xlabel("Train step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
def pt_adv_main1_add_digital(diglength):
    '''
    This program is used to test the influence of digital length 
    '''
    step = 500
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The length of digital is :", diglength)
    model = myAdvPTRNNModel(mod = "RNN").to(cuda0)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(step, model, optimizer, cuda0, diglength)
    for i in range(0, int(step/10)):
        if model.accuracy[i] == 1:
            break
    print("When step = ", i*10, "Accuracy = 1")

    plt.subplot(2, 1, 1)
    StepLos = np.arange(0, step)
    plt.plot(StepLos, model.loss)
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.subplot(2, 1, 2)
    StepAcc = np.arange(0, step, step = 10)
    plt.plot(StepAcc, model.accuracy)
    plt.xlabel("Train step")
    plt.ylabel("Accuracy")
    plt.show()
    
def pt_adv_main2_change_dig():
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    step = 600
    diglength = 10
    los, acc = [], []
    Steplos, Stepacc = np.arange(0, step, 1), np.arange(0, step, 10)
    for i in range(20, 101, 10):
        print("The dim of embedding layer is :", i)
        model = myAdvPTRNNModel(dim = i, mod = "RNN").to(cuda0)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        train(step, model, optimizer, cuda0, diglength)
        los.append(model.loss)
        acc.append(model.accuracy)
        for j in range(0, int(step/10)):
            if model.accuracy[j] == 1:
                break
        print("When step = ", j*10, "Accuracy = 1")
    plt.subplot(2, 1, 1)    
    for i in range(0, 9):
        plt.plot(Steplos, los[i], label="dim="+str(i*10+20))
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.legend()    
    
    plt.subplot(2, 1, 2)
    for i in range(0, 9):
        plt.plot(Stepacc, acc[i], label="dim="+str(i*10+20))
    plt.xlabel("Train step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
def pt_adv_main3_change_rnn_layer():
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    step = 800
    diglength = 10
    los, acc = [], []
    Steplos, Stepacc = np.arange(0, step, 1), np.arange(0, step, 10)
    for i in range(1, 5):
        print("The dim of embedding layer is :", i)
        model = myAdvPTRNNModel(layers = i, mod = "RNN").to(cuda0)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        train(step, model, optimizer, cuda0, diglength)
        los.append(model.loss)
        acc.append(model.accuracy)
        for j in range(0, int(step/10)):
            if model.accuracy[j] == 1:
                break
        print("When step = ", j*10, "Accuracy = 1")
    plt.subplot(2, 1, 1)    
    for i in range(0, 4):
        plt.plot(Steplos, los[i], label="rnn-"+str(i+1))
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.legend()    
    
    plt.subplot(2, 1, 2)
    for i in range(0, 4):
        plt.plot(Stepacc, acc[i], label="rnn-"+str(i+1))
    plt.xlabel("Train step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
def pt_adv_main4_change_rnn_to_gru():
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    step = 600
    diglength = 10
    los, acc = [], []
    Steplos, Stepacc = np.arange(0, step, 1), np.arange(0, step, 10)
    for i in range(1, 5):
        print("The dim of embedding layer is :", i)
        model = myAdvPTRNNModel(mod = "GRU").to(cuda0)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        train(step, model, optimizer, cuda0, diglength)
        los.append(model.loss)
        acc.append(model.accuracy)
        for j in range(0, int(step/10)):
            if model.accuracy[j] == 1:
                break
        print("When step = ", j*10, "Accuracy = 1")
    plt.subplot(2, 1, 1)    
    for i in range(0, 4):
        plt.plot(Steplos, los[i], label="GRU-"+str(i+1))
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.legend()    
    
    plt.subplot(2, 1, 2)
    for i in range(0, 4):
        plt.plot(Stepacc, acc[i], label="GRU-"+str(i+1))
    plt.xlabel("Train step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
def pt_adv_main5_change_rnn_to_lstm():
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    step = 600
    diglength = 10
    los, acc = [], []
    Steplos, Stepacc = np.arange(0, step, 1), np.arange(0, step, 10)
    for i in range(1, 5):
        print("The dim of embedding layer is :", i)
        model = myAdvPTRNNModel(mod = "LSTM").to(cuda0)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        train(step, model, optimizer, cuda0, diglength)
        los.append(model.loss)
        acc.append(model.accuracy)
        for j in range(0, int(step/10)):
            if model.accuracy[j] == 1:
                break
        print("When step = ", j*10, "Accuracy = 1")
    plt.subplot(2, 1, 1)    
    for i in range(0, 4):
        plt.plot(Steplos, los[i], label="LSTM-"+str(i+1))
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.legend()    
    
    plt.subplot(2, 1, 2)
    for i in range(0, 4):
        plt.plot(Stepacc, acc[i], label="LSTM-"+str(i+1))
    plt.xlabel("Train step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
def pt_adv_main6_change_lr1():
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    step = 600
    diglength = 10
    los, acc = [], []
    Steplos, Stepacc = np.arange(0, step, 1), np.arange(0, step, 10)
    for i in range(0, 3):
        lr_set = 10**(i-3)
        l=1
        print("torch.optim.Adam.lr is :", l,"*10^", i-3)
        model = myAdvPTRNNModel(mod = "RNN").to(cuda0)
        #print(lr_set)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=l*lr_set)
        train(step, model, optimizer, cuda0, diglength)
        los.append(model.loss)
        acc.append(model.accuracy)
        for j in range(0, int(step/10)):
            if model.accuracy[j] == 1:
                break
        print("When step = ", j*10, "Accuracy = 1")
        
        l=1.5
        print("torch.optim.Adam.lr is :", l,"*10^", i-3)
        model = myAdvPTRNNModel(mod = "RNN").to(cuda0)
        #print(lr_set)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=l*lr_set)
        train(step, model, optimizer, cuda0, diglength)
        los.append(model.loss)
        acc.append(model.accuracy)
        for j in range(0, int(step/10)):
            if model.accuracy[j] == 1:
                break
        print("When step = ", j*10, "Accuracy = 1")
        
    plt.subplot(2, 1, 1)    
    for i in range(0, 6):
        plt.plot(Steplos, los[i], label="lr="+str(1 if i % 2 == 0 else 1.5)+"*10^"+str(int(i/2)-3))
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.legend()    
    
    plt.subplot(2, 1, 2)
    for i in range(0, 6):
        plt.plot(Stepacc, acc[i], label="lr="+str(1 if i % 2 == 0 else 1.5)+"*10^"+str(int(i/2)-3))
    plt.xlabel("Train step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
def pt_adv_main7_change_lr2():
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    step = 150
    diglength = 10
    los, acc = [], []
    Steplos, Stepacc = np.arange(0, step, 1), np.arange(0, step, 10)
    for i in range(0, 6):
        lr_set = 0.01
        l=(i+5)/10
        print("torch.optim.Adam.lr is :", l,"*10^-2")
        model = myAdvPTRNNModel(mod = "RNN").to(cuda0)
        #print(lr_set)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=l*lr_set)
        train(step, model, optimizer, cuda0, diglength)
        los.append(model.loss)
        acc.append(model.accuracy)
        for j in range(0, int(step/10)):
            if model.accuracy[j] == 1:
                break
        print("When step = ", j*10, "Accuracy = 1")
    for i in range(1, 6):
        lr_set = 0.01
        l = i
        print("torch.optim.Adam.lr is :", l,"*10^-2")
        model = myAdvPTRNNModel(mod = "RNN").to(cuda0)
        #print(lr_set)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=l*lr_set)
        train(step, model, optimizer, cuda0, diglength)
        los.append(model.loss)
        acc.append(model.accuracy)
        for j in range(0, int(step/10)):
            if model.accuracy[j] == 1:
                break
        print("When step = ", j*10, "Accuracy = 1")
    
    plt.subplot(2, 1, 1)    
    for i in range(0, 10):
        plt.plot(Steplos, los[i], label="lr="+str((i+5)/10)+"*10^-2")
    plt.xlabel("Train step")
    plt.ylabel("Loss")
    plt.legend()    
    
    plt.subplot(2, 1, 2)
    for i in range(0, 10):
        plt.plot(Stepacc, acc[i], label="lr="+str((i+5)/10)+"*10^-2")
    plt.xlabel("Train step")
    plt.ylabel("Accuracy")
    plt.show()
def pt_adv_main():
    #pt_adv_main1_add_digital(200)
    #pt_adv_main2_change_dig()
    #pt_adv_main3_change_rnn_layer()
    #pt_adv_main4_change_rnn_to_gru()
    #pt_adv_main5_change_rnn_to_lstm()
    #pt_adv_main6_change_lr1()
    pt_adv_main7_change_lr2()
    '''
    cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diglength 
    model = myPTRNNModel().to(cuda0)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(1000, model, optimizer, cuda0, 10)
    print("Accuracy = ", evaluate(model, cuda0, 10))
    '''
