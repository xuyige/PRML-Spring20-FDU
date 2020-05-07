
import numpy as np
import torch
import torch.nn as nn
import random
import pdb
import os

from .data import prepare_batch, gen_data_batch, results_converter

batch_size = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2, batch_first = True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):

        num1_ = self.embed_layer(num1)
        num2_ = self.embed_layer(num2)
        nums = torch.cat([num1_,num2_], dim = -1)
        hiddens, h_n = self.rnn(nums)
        logits = self.dense(hiddens)

        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.lstm = nn.LSTM(input_size = 64, hidden_size = 64, num_layers =2, dropout = 0.5, batch_first = True )
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        num1_ = self.embed_layer(num1)
        num2_ = self.embed_layer(num2)
        nums = torch.cat([num1_,num2_], dim = -1)
        hiddens, (h_n, c_n) = self.lstm(nums)
        logits = self.dense(hiddens)
        return logits


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    return losses(logits.view(-1, 10), labels.view(-1))


def train_one_step(model, optimizer, x, y, label):
    
    optimizer.zero_grad()
    logits = model(torch.tensor(x).to(device), torch.tensor(y).to(device))
    loss = compute_loss(logits, torch.tensor(label).to(device))

    # compute gradient
    loss.backward()
    optimizer.step()
    return loss.item()

def train_one_step_adv(model, optimizer, x, y, label):
    
    optimizer.zero_grad()
    logits = model(x.to(device), y.to(device))
    loss = compute_loss(logits, label.to(device))

    # compute gradient
    loss.backward()
    optimizer.step()

    # compute acc
    pred = logits.argmax(axis=-1)
    length = pred.size(1)
    acc = ((pred.cpu() == label).sum(axis = -1) == length).float().mean()

    return loss.item(), acc.item()

def train(steps, model, optimizer, maxlen):
    model.train()
    loss = 0.0
    accuracy = 0.0
    model.to(device)
    for step in range(steps):
        datas = gen_data_batch(batch_size=batch_size, start=0, end=int(maxlen*'5'))
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=maxlen+1)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        if step % 50 == 0:
            print('step', step, ': loss', loss)

    return loss

def train_adv(model, optimizer, maxlen, train_dataset):
    model.train()

    datas = train_dataset
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=maxlen+1)
    Nums1 = torch.tensor(Nums1)
    Nums2 = torch.tensor(Nums2)
    results = torch.tensor(results)

    dataset_size = len(datas[0])

    random_map = random.sample(range(dataset_size), dataset_size)
    batch_list = [ random_map[i:i+batch_size] for i in range(0,max(1, dataset_size+1-batch_size),batch_size)]

    total_loss = 0.0
    total_acc = 0.0
    model.to(device)
    for i,batch in enumerate(batch_list):
        batch_size_ = len(batch)
        Nums1_ = Nums1[batch]
        Nums2_ = Nums2[batch]
        results_ = results[batch]
        
        loss, acc = train_one_step_adv(model, optimizer, Nums1_,
                              Nums2_, results_)
        total_loss += loss*batch_size_
        total_acc += acc*batch_size_

    total_loss = total_loss/dataset_size
    total_acc = total_acc/ dataset_size

    #print('Train: Accuracy is {0}, Loss is {1}'.format(total_acc, total_loss) )
    return 

def evaluate(model, maxlen, test_dataset):

    model.eval()
    
    datas = test_dataset
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=maxlen+1)
    Nums1 = torch.tensor(Nums1)
    Nums2 = torch.tensor(Nums2)

    dataset_size = len(datas[0])

    order_map = [i for i in range(dataset_size)]
    batch_list = [ order_map[i:i+batch_size] for i in range(0,dataset_size+1-batch_size,batch_size)]

    total_acc = 0
    for i,batch in enumerate(batch_list):
        batch_size_ = len(batch)
        Nums1_ = Nums1[batch]
        Nums2_ = Nums2[batch]
        with torch.no_grad():
            logits = model(Nums1_.to(device), Nums2_.to(device))
        logits = logits.cpu().numpy()
        pred = np.argmax(logits, axis=-1)
        res = results_converter(pred)
        accuracy = np.mean([o[0]==o[1] for o in zip(datas[2][i*batch_size: i*batch_size+batch_size_], res)])
        total_acc += accuracy * batch_size_

    print('Test: Accuracy is: ', total_acc / dataset_size)
    model.train()


def pt_main(arg):
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    test_dataset_size = 1000
    maxlen = arg.maxlen

    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    path = str(maxlen)+'.pt'

    if os.path.exists(path) and arg.load:
        model.load_state_dict(torch.load(path))
        print('parameters loaded.')

    test_dataset = gen_data_batch(batch_size=test_dataset_size, start=int(maxlen*'5'), end=int(maxlen*'9'))

    print('Maxlen: {0} Batch_size: {1} test_size:{2} seed:{3} load_sign:{4}'.format(maxlen, batch_size, test_dataset_size,seed, arg.load))
    for i in range(300):
        print(i,'th evaluation')
        train(100, model, optimizer, maxlen =maxlen)
        evaluate(model, maxlen =maxlen, test_dataset = test_dataset)
        torch.save(model.state_dict(),path)


def pt_adv_main(arg):
    if arg.rnn_adv:
        model = myAdvPTRNNModel()
    else:
        model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    test_dataset_size = 1000
    train_dataset_size = arg.train_dataset_size
    maxlen = arg.maxlen

    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    path = str(maxlen)+'.pt'

    if os.path.exists(path) and arg.load:
        model.load_state_dict(torch.load(path))
        print('parameters loaded.')

    test_dataset = gen_data_batch(batch_size=test_dataset_size, start=int(maxlen*'5'), end=int(maxlen*'9'))
    train_dataset = gen_data_batch(batch_size=train_dataset_size, start=0, end=int(maxlen*'5'))

    print('Maxlen: {0} Batch_size: {1} test_size:{2} seed:{3} load_sign Train_size:{4}'.format(maxlen, batch_size, test_dataset_size,seed, arg.load, arg.train_dataset_size))
    for i in range(20000):
        
        train_adv(model, optimizer, maxlen = maxlen, train_dataset= train_dataset) 
        # python中，如果参数是“不可变对象” - 数，字符串，元祖，则相当于传值；如果参数是列表、字典，则相当于传引用
        if(i % max(1,int(1000/arg.train_dataset_size)) == 0): # 如果数据集大小100 则10个epoch验证一次；数据集大小1000，1个epoch一次；
            print(i,'th Epoch')
            evaluate(model, maxlen =maxlen, test_dataset = test_dataset)
        torch.save(model.state_dict(),path)