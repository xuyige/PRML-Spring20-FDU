import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .data import prepare_batch, gen_data_batch, results_converter


class myPTRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        self.embed_layer = nn.Embedding(10, 32)
        # nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity=tanh, bias=True, batch_first=False, dropout=0, bidirectional=False)
        self.rnn = nn.RNN(64, 64, 2, batch_first=True)
        # nn.Linear（in_features，out_features, bias=True ）
        self.dense = nn.Linear(64, 10)
        self.loss_record = []
        self.acc_record = []

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1_embedded = self.embed_layer(num1)
        num2_embedded = self.embed_layer(num2)
        num = torch.cat((num1_embedded, num2_embedded), -1)
        out, h_n = self.rnn(num)
        logits = self.dense(out)
        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self, layers=2, model="rnn"):
        '''
        Please finish your code here.
        '''
        super().__init__()
        self.embed_layer = nn.Embedding(10, 32)
        if model == "rnn":
            self.rnn = nn.RNN(64, 64, layers, batch_first=True)
        elif model == "lstm":
            self.rnn = nn.LSTM(64, 64, layers, batch_first=True)
        elif model == "gru":
            self.rnn = nn.GRU(64, 64, layers, batch_first=True)
        self.dense = nn.Linear(64, 10)
        self.loss_record = []
        self.acc_record = []

    def forward(self, num1, num2):
        '''
        Please finish your code here.
        '''
        num1_embedded = self.embed_layer(num1)
        num2_embedded = self.embed_layer(num2)
        num = torch.cat((num1_embedded, num2_embedded), -1)
        out, h_n = self.rnn(num)
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


def train(steps, model, optimizer, start=0, mid=555555555, end=999999999):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=start, end=mid)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=19)
        loss = train_one_step(model, optimizer, Nums1,
                              Nums2, results)
        if step % 50 == 0:
            print('step', step, ': loss', loss)
            model.loss_record.append(loss)
            acc = evaluate(model, start=mid, end=end)
            model.acc_record.append(acc)
    return loss


def evaluate(model, start=555555555, end=999999999):
    datas = gen_data_batch(batch_size=2000, start=start, end=end)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=19)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1), torch.tensor(Nums2))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    # for o in list(zip(datas[2], res))[:20]:
    #     print(o[0], o[1], o[0]==o[1])
    acc = np.mean([o[0] == o[1] for o in zip(datas[2], res)])
    print('accuracy is: %g' % acc)
    return acc


def draw(model_list, title, legend_list=None):
    loss_list = []
    acc_list = []
    for model in model_list:
        plt.subplot(121)
        loss, = plt.plot(model.loss_record)
        plt.xlabel('iter/50:')
        plt.ylabel('loss:')
        loss_list.append(loss)
    plt.title(title + '-loss')
    if legend_list is not None:
        plt.legend(loss_list, legend_list, loc='upper right')
    for model in model_list:
        plt.subplot(122)
        acc, = plt.plot(model.acc_record)
        plt.xlabel('iter/50:')
        plt.ylabel('accuracy:')
        acc_list.append(acc)
    plt.title(title + '-accuracy')
    if legend_list is not None:
        plt.legend(acc_list, legend_list, loc='lower right')
    plt.savefig(title)
    plt.clf()


def compare_length():
    """
    compare the result with different max_length of digit
    :return: nothing
    """
    print("Test different data lengths!")
    model_list = []
    legend_list = []
    for len in range(1, 19, 3):
        model = myPTRNNModel()
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        train(1000, model, optimizer, start=0, mid=int(10 ** len), end=int(2 * 10 ** len))
        model_list.append(model)
        legend_list.append("len:" + str(len))
    draw(model_list, title="DataLength", legend_list=legend_list)


def compare_model():
    """
    compare the result of different models.
    :return:
    """
    print("Test different models!")
    model_list = []
    type_list = ['rnn', 'lstm', 'gru']
    legend_list = []
    for type in type_list:
        print(type)
        model = myAdvPTRNNModel(model=type)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        train(1000, model, optimizer)
        model_list.append(model)
        legend_list.append(type)
    draw(model_list, title="DifferentModels", legend_list=legend_list)


def compare_learning_rate():
    """
    compare the results with different learning rates.
    :return:
    """
    print("Test different learning rates!")
    model_list = []
    legend_list = []
    learning_rates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    for learning_rate in learning_rates:
        model = myPTRNNModel()
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        train(1000, model, optimizer)
        model_list.append(model)
        legend_list.append("lr:" + str(learning_rate))
    draw(model_list, "LearningRate", legend_list)


def compare_layers():
    """
    compare the result with different amounts of layers.
    :return:
    """
    print("Test different layers!")
    model_list = []
    legend_list = []
    for layer in range(1, 5):
        model = myAdvPTRNNModel(layers=layer)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        train(1000, model, optimizer)
        model_list.append(model)
        legend_list.append("layers:" + str(layer))
    draw(model_list, "Layers", legend_list)


def get_best(learning_rate, layers, model, data_length):
    model = myAdvPTRNNModel(layers=layers, model=model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    train(1000, model, optimizer, start=0, mid=int(10 ** data_length), end=int(2 * 10 ** data_length))
    draw([model], "TheBest")


def pt_main():
    model = myPTRNNModel()
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    train(3000, model, optimizer)
    evaluate(model)
    draw([model], "RNN")


def pt_adv_main():
    '''
    Please finish your code here.
    '''
    compare_length()
    compare_model()
    compare_learning_rate()
    compare_layers()
    get_best(0.01, 3, 'gru', 16)
