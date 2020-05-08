
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data import prepare_batch, gen_data_batch, results_converter


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = []
        self.accuracy = []
        self.embed_layer = nn.Embedding(10, 32)     # 初始化一个10 * 32的随机矩阵,将10维的数据可以映射到32维
        self.rnn = nn.RNN(64, 64, 2, batch_first = True)    # 输入维度64，隐藏层神经元数64，网络层数2
        self.dense = nn.Linear(64, 10)  # 线性层 y = Wx + b

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)
        # print(num1)
        # 数据连接
        input = torch.cat((num1, num2), 2)
        output, hidden = self.rnn(input)
        logits = self.dense(output)

        return logits


class myAdvPTRNNModel(nn.Module):
    # def __init__(self):
    def __init__(self, n_layer = 2, model = "LSTM"):
        '''
        Please finish your code here.
        '''
        super().__init__()

        self.loss = []
        self.accuracy = []

        self.embed_layer = nn.Embedding(10, 32)
        if model == "LSTM":
            self.rnn = nn.LSTM(64, 64, n_layer, batch_first = True)
        if model=="GRU":
            self.rnn = nn.GRU(64, 64, n_layer, batch_first = True)
        else:
            self.rnn = nn.RNN(64, 64, n_layer, batch_first = True)

        self.dense = nn.Linear(64, 10)

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)
        # print(num1)
        # 数据连接
        input = torch.cat((num1, num2), 2)
        output, hidden = self.rnn(input)
        logits = self.dense(output)

        return logits


def compute_loss(logits, labels):
    # nn.CrossEntropyLoss()是nn.logSoftmax()和nn.NLLLoss()的整合
    losses = nn.CrossEntropyLoss()
    # view(x, y)调整维度为(x * y), -1表示自适应
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


def train(steps, model, optimizer, max_len = 10, test_len = 10, b_size = 20):
    loss = 0.0
    accuracy = 0.0
    end_num = 10 ** (max_len - 1)
    for step in range(steps):
        # [0, 555555555)的随机长度为200的数组
        datas = gen_data_batch(batch_size=b_size, start=0, end=(end_num))
        # 获得200 * maxlen的二维数组
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=max_len + 1)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        model.loss.append(loss)
        # acc = evaluate(model, test_len)
        # model.accuracy.append(acc)
        if step % 50 == 0:
            acc = evaluate(model, test_len)
            model.accuracy.append(acc)
            print('step', step, '\tloss', loss, '\taccuracy', acc)

    return loss


def evaluate(model, max_len = 10):
    start_num = 0
    end_num = 10 ** (max_len - 1) - 1

    datas = gen_data_batch(batch_size=20, start=start_num, end=end_num)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=max_len + 1)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])
    acc = np.mean([o[0]==o[1] for o in zip(datas[2], res)])
    # print('accuracy is: %g' % acc)
    return acc

def draw(model):
    plt.subplot(2, 1, 1)
    plt.plot(model.loss)
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(model.accuracy)
    plt.ylabel('Accuracy')
    plt.show()

def all_draw(diff):
    plt.subplot(2, 1, 1)
    i = 0
    for l in all_loss:
        plt.plot(l, linewidth = '1', label = diff[i])
        i += 1
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    i = 0
    for a in all_accuracy:
        plt.plot(a, linewidth = '1', label = diff[i])
        i += 1
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

def pt_main(n_train = 3000, max_len = 5):
    model = myPTRNNModel()
    # 构建优化器对象，lr为步长因子
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    train(n_train, model, optimizer, max_len, max_len)
    # evaluate(model)
    draw(model)

all_loss = []
all_accuracy = []
def pt_adv_main(n_train = 3000, max_len = 10, test_len = 10, n_layer = 3, typ = "LSTM", my_lr = 0.01, b_size = 20):
    '''
    Please finish your code here.
    '''
    model = myAdvPTRNNModel(n_layer, typ)
    # 构建优化器对象，lr为步长因子
    optimizer = torch.optim.Adam(params=model.parameters(), lr=my_lr)

    train(n_train, model, optimizer, max_len, test_len, b_size)
    # evaluate(model)
    draw(model)
    all_loss.append(model.loss)
    all_accuracy.append(model.accuracy)

def test_1():
    pt_adv_main(3000, 1, 10, 2, "RNN", 0.001)
    pt_adv_main(3000, 2, 10, 2, "RNN", 0.001)
    pt_adv_main(3000, 3, 10, 2, "RNN", 0.001)
    pt_adv_main(3000, 5, 10, 2, "RNN", 0.001)
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.001)
    diff = ['1', '2', '3', '5', '10']

    all_draw(diff)

def test_2():
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.001, 1)
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.001, 5)
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.001, 10)
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.001, 100)
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.001, 1000)
    diff = ['1', '5', '10', '100', '1000']

    all_draw(diff)

def test_3():
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.00001, 10)
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.0001, 10)
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.001, 10)
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.01, 10)
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.1, 10)
    diff = ['0.00001', '0.0001', '0.001', '0.01', '0.1']

    all_draw(diff)

def test_4():
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.001, 10)
    pt_adv_main(3000, 10, 10, 2, "GRU", 0.001, 10)
    pt_adv_main(3000, 10, 10, 2, "LSTM", 0.001, 10)

    diff = ['RNN', 'GRU', 'LSTM']

    all_draw(diff)

def test_5():
    diff = ['1', '2', '3', '4','1', '2', '3', '4','1', '2', '3', '4']

    pt_adv_main(3000, 10, 10, 1, "RNN", 0.001, 10)
    pt_adv_main(3000, 10, 10, 2, "RNN", 0.001, 10)
    pt_adv_main(3000, 10, 10, 3, "RNN", 0.001, 10)
    pt_adv_main(3000, 10, 10, 4, "RNN", 0.001, 10)
 

    pt_adv_main(3000, 10, 10, 1, "GRU", 0.001, 10)
    pt_adv_main(3000, 10, 10, 2, "GRU", 0.001, 10)
    pt_adv_main(3000, 10, 10, 3, "GRU", 0.001, 10)
    pt_adv_main(3000, 10, 10, 4, "GRU", 0.001, 10)


    pt_adv_main(3000, 10, 10, 1, "LSTM", 0.001, 10)
    pt_adv_main(3000, 10, 10, 2, "LSTM", 0.001, 10)
    pt_adv_main(3000, 10, 10, 3, "LSTM", 0.001, 10)
    pt_adv_main(3000, 10, 10, 4, "LSTM", 0.001, 10)
    all_draw(diff)


if __name__ == '__main__':
    pt_main()
    pt_adv_main(300, 10, 10, 3, "RNN", 0.01, 1000)
    # test_1()
    # test_2()
    # test_3()
    # test_4()
    # test_5()