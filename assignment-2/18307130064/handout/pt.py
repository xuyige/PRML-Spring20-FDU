
import numpy as np
import torch
import torch.nn as nn

from .gen import prepare_batch, gen_data_batch, results_converter,gen_data_batch2


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 3,batch_first=True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        logits1 = self.embed_layer(num1)
        logits2 = self.embed_layer(num2)
        logits = torch.cat([logits1,logits2],dim=2)
        logits,hn = self.rnn(logits,None)
        logits = self.dense(logits)
        return logits



class myAdvPTRNNModel(nn.Module):
    def __init__(self):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.embed_layer = nn.Embedding(10, 64)
        self.rnn = nn.RNN(128, 128, 2,batch_first=True)
        self.dense = nn.Linear(128, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        logits1 = self.embed_layer(num1)
        logits2 = self.embed_layer(num2)
        logits = torch.cat([logits1,logits2],dim=2)
        logits,hn = self.rnn(logits,None)
        logits = self.dense(logits)
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
    datas = gen_data_batch(batch_size=2000, start=555555555, end=999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    #for o in list(zip(datas[2], res))[:20]:
    #    print(o[0],"@@", o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))

def train2 ( steps , model , optimizer ):
    loss = 0.0
    accuracy = 0.0
    for step in range ( steps ):
        nums1,nums2,results = gen_data_batch2 ( batch_size=200 , len=4 )
        loss = train_one_step ( model , optimizer , nums1 , nums2 , results )
        if step % 50 == 0:
            print ( 'step',step,':loss',loss)
    return loss

def evaluate2(model):
    batch_size = 2000
    len = 100
    nums1,nums2,results = gen_data_batch2 ( batch_size = batch_size , len = len )
    with torch.no_grad():
        logits = model ( torch.tensor(nums1) , torch.tensor(nums2) )
    logits = logits.numpy ()
    pred = np.argmax ( logits , axis = -1 )
    acc = 0
    for i in range(batch_size):
        acc += 1
        for j in range(len):
            if pred[i][j] != results[i][j]:
                acc -= 1
                break
    for i in range(len):
        print ( pred[0][i] ,end='' )
    print ()
    for i in range(len):
        print ( results[0][i] ,end='' )
    print ()
    print ( 'accuracy is:' , acc / batch_size )

def pt_main():
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    #train(3000, model, optimizer)
    train(1000, model, optimizer)
    evaluate(model)


def pt_adv_main():
    model = myAdvPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=0.001)
    train2 ( 1000 , model , optimizer )
    evaluate2 ( model )
