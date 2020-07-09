import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from data import prepare_batch, gen_data_batch, results_converter
else:
    from .data import prepare_batch, gen_data_batch, results_converter


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.acc = []
        self.embed_layer = nn.Embedding(10, 32)
        self.rnn = nn.RNN(64, 64, 2, batch_first=True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)
        temp = torch.cat((num1, num2), 2)
        output, _ = self.rnn(temp)
        logits = self.dense(output)
        
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self, layers=2, model='RNN'):
        '''
        Please finish your code here.
        '''
        super().__init__()
        
        self.losses = []
        self.acc = []
        self.embed_layer = nn.Embedding(10, 32)
        m2m = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        self.rnn = m2m[model](64, 64, layers, batch_first=True)
        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)
        temp = torch.cat((num1, num2), 2)
        output, _ = self.rnn(temp)
        logits = self.dense(output)
        
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


def train(steps, model, optimizer, maxlen=9):
    loss = 0.0
    accuracy = 0.0
    end_num = 10**maxlen
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=0, end=end_num)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=maxlen + 2)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        model.losses.append(loss) # added
        if step % 50 == 0:
            print('step', step, ': loss', loss)
            acc = evaluate(model)
            model.acc.append(acc)

    return loss


def evaluate(model, maxlen=9, batch_size=1000):
    end_num = 10**maxlen
    datas = gen_data_batch(batch_size=batch_size, start=0, end=end_num)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=maxlen + 2)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])

    acc = np.mean([o[0]==o[1] for o in zip(datas[2], res)]) # added
    print('accuracy is: %g' % acc)
    return acc

def plot(losses, acc, model='RNN', title=None): # , layers=2, maxlen=9, lr=0.01):
    # plot losses
    x = np.arange(len(losses))
    label = f'{model} loss'
    if None != title:
        label = title + label
    plt.plot(x, losses, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.savefig(f'{model}-loss.png')
    plt.close()
    
    # plot accuracy
    x = np.arange(len(acc))
    x = x * 50
    label = f'{model} Accuracy'
    if None != title:
        label = title + label
    plt.plot(x, acc, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{model}-acc.png')
    plt.close()

def plot_single_graph(x, y, title=None, labels=None, xlabel=None, ylabel=None, file=None):
    for idx, i in enumerate(y):
        plt.plot(x, i, label=labels[idx])
        if None != xlabel:
            plt.xlabel(xlabel)
        if None != ylabel:
            plt.ylabel(ylabel)
        if None != title:
            plt.title(title)
    plt.legend()
    if None == file:
        plt.show()
    else:
        plt.savefig(file)
    plt.close()

def pt_main(lr=0.001):
    model = myPTRNNModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train(2000, model, optimizer)
    evaluate(model)
    plot(model.losses, model,acc)

def pt_adv_main(m='RNN', maxlen=9, epochs=2000, layers=2, lr=0.001):
    '''
    Please finish your code here.
    '''
    model = myAdvPTRNNModel(model=m, layers=layers)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train(epochs, model, optimizer, maxlen=maxlen)
    evaluate(model, maxlen=maxlen)
    return model.losses, model.acc

def train_test(case=1):
    losses, accs = [], []
    if case == 1:
        # layers
        lb = [i for i in range(1, 5)]
        for i in lb:
            loss, acc = pt_adv_main(epochs=500, layers=i, maxlen=14)
            losses.append(loss)
            accs.append(acc)
        x = np.arange(len(losses[0]))
        plot_single_graph(x, losses, labels=lb, title="Different Layers with RNN Model", xlabel="Iteration", ylabel="Loss Value", file="test-layers-loss.png")
        x = np.arange(len(accs[0]))
        x = x * 50
        plot_single_graph(x, accs, labels=lb, title="Different Layers with RNN Model", xlabel="Iteration", ylabel="Accuracy", file="test-layers-acc.png")

    if case == 2:
        # max number length
        nums = [10, 20, 50, 100, 200]
        for i in nums:
            loss, acc = pt_adv_main(epochs=1000, layers=2, maxlen=i)
            losses.append(loss)
            accs.append(acc)
        x = np.arange(len(losses[0]))
        plot_single_graph(x, losses, labels=nums, title="Different Max Number Length with RNN Model", xlabel="Iteration", ylabel="Loss Value", file="test-nums-loss.png")
        x = np.arange(len(accs[0]))
        x = x * 50
        plot_single_graph(x, accs, labels=nums, title="Different Max Number Length with RNN Model", xlabel="Iteration", ylabel="Accuracy", file="test-nums-acc.png")
    
    if case == 3:
        # learning rate
        lrs = [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1]
        for i in lrs:
            loss, acc = pt_adv_main(epochs=500, layers=2, lr=i)
            losses.append(loss)
            accs.append(acc)
        x = np.arange(len(losses[0]))
        title = "Learning Rate with RNN Model"
        plot_single_graph(x, losses, labels=lrs, title=title, xlabel="Iteration", ylabel="Loss Value", file="test-lrs-loss.png")
        x = np.arange(len(accs[0]))
        x = x * 50
        plot_single_graph(x, accs, labels=lrs, title=title, xlabel="Iteration", ylabel="Accuracy", file="test-lrs-acc.png")

    if case == 4:
        # different models
        models = ['RNN', 'LSTM', 'GRU']
        for i in models:
            loss, acc = pt_adv_main(m=i, epochs=500, layers=2)
            losses.append(loss)
            accs.append(acc)
        x = np.arange(len(losses[0]))
        title = "Different Models"
        plot_single_graph(x, losses, labels=models, title=title, xlabel="Iteration", ylabel="Loss Value", file="test-models-loss.png")
        x = np.arange(len(accs[0]))
        x = x * 50
        plot_single_graph(x, accs, labels=models, title=title, xlabel="Iteration", ylabel="Accuracy", file="test-models-acc.png")

if __name__ == '__main__':
    train_test(case=4)
