import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from data import prepare_batch, gen_data_batch, results_converter, gen_data_batch_longer


class myPTRNNModel(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=2, batch_size=200,
                 model_device=torch.device("cpu")):
        super().__init__()
        # parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        # model
        self.embed_layer = nn.Embedding(10, int(self.hidden_size / 2))
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity='relu')
        self.dense = nn.Linear(self.hidden_size, 10)
        self.h = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(model_device)
        # record
        self.accuracy_record = []
        self.loss_record = []

    def forward(self, num1, num2):
        """
        :param num1: shape(batch_size, maxlen)
        :param num2: shape(batch_size, maxlen)
        :return: logits: shape(batch_size, maxlen)
        """
        # Embedding
        # (batch_size, maxlen, embedding_dim)
        num1_ebd = self.embed_layer(num1)
        num2_ebd = self.embed_layer(num2)

        # Rnn
        # input: (seq_len=maxlen, batch, input_size=2 * embedding_dim)
        input_num = torch.cat((num1_ebd, num2_ebd), dim=2).transpose(1, 0)
        output, self.h = self.rnn(input_num, self.h)

        # Densing
        # logits: (seq_len, batch, 10)
        logits = self.dense(output).transpose(1, 0)

        return logits


class myAdvPTRNNModel(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=2, batch_size=200, mode='rnn', nonlinear='relu',
                 model_device=torch.device("cpu"), h_initial='randn', h_not='True'):
        '''
        Please finish your code here.
        '''
        super().__init__()
        # parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode
        self.nonlinear = nonlinear
        self.batch_size = batch_size
        self.h_initial = h_initial
        self.h_not = h_not

        # model
        self.embed_layer = nn.Embedding(10, int(self.hidden_size / 2))
        if self.mode == 'rnn':
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity=self.nonlinear)
            if self.h_initial == 'randn':
                self.h = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(model_device)
            elif self.h_initial == 'randn10':
                self.h = 10 * torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(model_device)
            elif self.h_initial == 'zero':
                self.h = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(model_device)
            elif self.h_initial == 'xavier':
                tensor = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(model_device)
                self.h = torch.nn.init.xavier_uniform_(tensor, gain=1)
            elif self.h_initial == 'xavier_he':
                tensor = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(model_device)
                self.h = torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu')
        elif self.mode == 'lstm':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
            self.h = (torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device),
                      torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device))
        elif self.mode == 'gru':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
            self.h = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)

        self.dense = nn.Linear(self.hidden_size, 10)

        # record
        self.accuracy_record = []
        self.loss_record = []

    def forward(self, num1, num2):
        """
        :param num1: shape(batch_size, maxlen)
        :param num2: shape(batch_size, maxlen)
        :return: logits: shape(batch_size, maxlen)
        """
        # Embedding
        # (batch_size, maxlen, embedding_dim)
        num1_ebd = self.embed_layer(num1)
        num2_ebd = self.embed_layer(num2)

        # Rnn
        # input: (seq_len=maxlen, batch, input_size=2 * embedding_dim)
        input_num = torch.cat((num1_ebd, num2_ebd), dim=2).transpose(1, 0)
        if self.h_not == 'True':
            output, self.h = self.rnn(input_num, self.h)
        elif self.h_not == 'False':
            output, self.h = self.rnn(input_num)
        else:
            zeros = torch.zeros_like(self.h)
            output, self.h = self.rnn(input_num, zeros)

        # Densing
        # logits: (seq_len, batch, 10)
        logits = self.dense(output).transpose(1, 0)

        return logits


def compute_loss(logits, labels):
    losses = nn.CrossEntropyLoss()
    # revise view -> reshape
    return losses(logits.reshape(-1, 10), labels.view(-1))


def train_one_step(model, optimizer, scheduler, x, y, label, device):
    model.train()
    optimizer.zero_grad()
    logits = model(torch.tensor(x).to(device), torch.tensor(y).to(device))
    loss = compute_loss(logits, torch.tensor(label).to(device))

    # compute gradient
    loss.backward(retain_graph=True)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.item()


def train(steps, model, optimizer, scheduler, evaluate_step, device, maxlen):
    loss = 0.0
    accuracy = 0.0
    for step in range(steps):
        # datas = gen_data_batch(batch_size=200, start=0, end=int((10 ** (maxlen - 1) - 1) / 9 * 5))
        datas = gen_data_batch_longer(batch_size=200, start=0, end=0.5, digitlen=maxlen - 1)

        Nums1, Nums2, results = prepare_batch(*datas, maxlen + 1)
        loss = train_one_step(model, optimizer, scheduler, Nums1,
                              Nums2, results, device)
        if step % 50 == 0:
            print('step', step, ': loss', loss)
            model.loss_record.append(loss)
        if step % evaluate_step == 0:
            accuracy = evaluate(model, device, maxlen)
            print('step', step, ':accuracy', accuracy)
            model.accuracy_record.append(accuracy)
    return loss


def evaluate(model, device, maxlen):
    # datas = gen_data_batch(batch_size=200, start=int((10 ** (maxlen - 1) - 1) / 9 * 5),
    #                        end=int((10 ** (maxlen - 1) - 1)))
    datas = gen_data_batch_longer(batch_size=200, start=0.5, end=1.0, digitlen=maxlen - 1)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen + 1)
    with torch.no_grad():
        logits = model(torch.tensor(Nums1).to(device), torch.tensor(Nums2).to(device))
    logits = logits.cpu().numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    #     for o in list(zip(datas[2], res))[:20]:
    #         print(o[0], o[1], o[0]==o[1])
    #     print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))
    return np.mean([o[0] == o[1] for o in zip(datas[2], res)])


def visualize(model_list, title, legend_list=None, loss_step=50, evaluate_step=50):
    loss_list = []
    accu_list = []
    for model in model_list:
        plt.subplot(121)
        p_loss, = plt.plot(model.loss_record)
        plt.xlabel('Iter/%d:' % loss_step)
        plt.ylabel('Loss:')
        loss_list.append(p_loss)

        plt.subplot(122)
        p_accu, = plt.plot(model.accuracy_record)
        plt.xlabel('Iter/%d:' % evaluate_step)
        plt.ylabel('Accuracy:')
        accu_list.append(p_accu)
    plt.subplot(121)
    plt.title(title + ':Loss')
    if legend_list is not None:
        plt.legend(loss_list, legend_list, loc='upper right')
    plt.subplot(122)
    plt.title(title + ':Accuracy')
    if legend_list is not None:
        plt.legend(accu_list, legend_list, loc='lower right')
    plt.show()


def visualize_list(loss_records, accuracy_records, title, legend_list=None, loss_step=50, evaluate_step=50):
    loss_list = []
    accu_list = []

    plt.subplot(121)
    for loss_record in loss_records:
        p_loss, = plt.plot(loss_record)
        plt.xlabel('Iter/%d:' % loss_step)
        plt.ylabel('Loss:')
        loss_list.append(p_loss)
    plt.title(title + ':Loss')
    if legend_list is not None:
        plt.legend(loss_list, legend_list, loc='upper right')

    plt.subplot(122)
    for accuracy_record in accuracy_records:
        p_accu, = plt.plot(accuracy_record)
        plt.xlabel('Iter/%d:' % evaluate_step)
        plt.ylabel('Accuracy:')
        accu_list.append(p_accu)
    plt.title(title + ':Accuracy')
    if legend_list is not None:
        plt.legend(accu_list, legend_list, loc='lower right')

    plt.show()


def pt_main(device=torch.device("cpu")):
    """
    默认前提下运行PTRNNModel
    """
    maxlen = 10
    evaluate_step = 50
    model = myPTRNNModel(num_layers=3, model_device=device).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005)
    train(500, model, optimizer, None, evaluate_step, device, maxlen)
    visualize([model], 'RNN')
    evaluate(model, device, maxlen)


def pt_adv_main(device=torch.device("cpu")):
    """
    默认前提下运行AdvPTRNNModel
    """
    milestones = [150, 300]
    evaluate_step = 50
    maxlen = 10
    lr = 0.025
    gamma = 0.2
    model = myAdvPTRNNModel(model_device=device).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    train(500, model, optimizer, scheduler, evaluate_step, device, maxlen)
    visualize([model], "test")
    evaluate(model, device, maxlen)


def experiment_maxlen(device):
    """
    maxlen 对于模型的影响
    """
    model_list = []
    legend_list = []
    repeat = 5
    loss_list = []
    accuracy_list = []
    maxlens = [5, 10, 25, 50, 70]
    evaluate_step = 50
    title = "Maxlen"
    for maxlen in maxlens:
        loss = 0
        accuracy = 0
        for i in range(repeat):
            model = myPTRNNModel(model_device=device).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
            train(500, model, optimizer, None, evaluate_step, device, maxlen)
            loss += np.array(model.loss_record)
            accuracy += np.array(model.accuracy_record)
        loss_list.append(loss / repeat)
        accuracy_list.append(accuracy / repeat)
        legend_list.append(title + "=" + str(maxlen))
    visualize_list(loss_list, accuracy_list, title, legend_list, evaluate_step=evaluate_step)


def experiment_layer(device):
    """
    RNN层数 对于模型学习的影响
    """
    maxlen = 5
    title = 'Num of Layers'
    model_list = []
    legend_list = []
    repeat = 5
    loss_list = []
    accuracy_list = []
    num_layers = [1, 2, 3, 4]
    evaluate_step = 50
    for num_layer in num_layers:
        loss = 0
        accuracy = 0
        for i in range(repeat):
            model = myAdvPTRNNModel(num_layers=num_layer, model_device=device).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
            train(1000, model, optimizer, None, evaluate_step, device, maxlen)
            loss += np.array(model.loss_record)
            accuracy += np.array(model.accuracy_record)
        loss_list.append(loss / repeat)
        accuracy_list.append(accuracy / repeat)
        # model_list.append(model)
        legend_list.append(title + "=" + str(num_layer))
    visualize_list(loss_list, accuracy_list, title, legend_list, evaluate_step=evaluate_step)
    # visualize(model_list, title, legend_list)


def experiment_lr(device):
    """
    学习率对于模型的影响
    """
    maxlen = 5
    evaluate_step = 50
    model_list = []
    legend_list = []
    lrs = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5]
    title = "LR"
    for lr in lrs:
        model = myPTRNNModel(model_device=device).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        train(1000, model, optimizer, None, evaluate_step, device, maxlen)
        model_list.append(model)
        legend_list.append(title + "=" + str(lr))

    milestones = [100]
    model = myPTRNNModel(model_device=device).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5)
    train(1000, model, optimizer, scheduler, evaluate_step, device, maxlen)
    model_list.append(model)
    legend_list.append(title + "=changing")
    visualize(model_list, title, legend_list)


def experiment_layer_h_not(device):
    """
    探索layer和h_not对模型学习的共同影响
    """
    lr = 0.001
    evaluate_step = 50
    maxlen = 10
    hs = ['True', 'False', 'Zero']
    num_layers = [1, 2, 3, 4, 5]
    datas = []
    for h in hs:
        data_list = []
        for num_layer in num_layers:
            model = myAdvPTRNNModel(num_layers=num_layer, h_not=h, model_device=device).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
            train(700, model, optimizer, None, evaluate_step, device, maxlen)
            if 1.0 in model.accuracy_record:
                iter_accuracy = (model.accuracy_record.index(1.0) + 1) * evaluate_step
            else:
                iter_accuracy = (len(model.accuracy_record)) * evaluate_step
            data_list.append(iter_accuracy)
        datas.append(data_list)
    sns.heatmap(np.array(datas), xticklabels=num_layers, yticklabels=hs)
    plt.title('Iteration when Accuracy comes to 1.0')
    plt.xlabel('NumofLayers')
    plt.ylabel('H_iteration')
    plt.show()
    # datas:(#lr, #layers)


def experiment_maxlen_lr(device):
    """
    探索layer和learning_rate对模型学习的共同影响
    """
    evaluate_step = 50
    maxlens = [5, 10, 25, 50, 70]
    lrs = [0.001, 0.002, 0.005, 0.01, 0.05]
    datas = []
    for lr in lrs:
        data_list = []
        for maxlen in maxlens:
            model = myPTRNNModel(num_layers=3, model_device=device).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
            train(500, model, optimizer, None, evaluate_step, device, maxlen)
            if 1.0 in model.accuracy_record:
                iter_accuracy = (model.accuracy_record.index(1.0) + 1) * evaluate_step
            else:
                iter_accuracy = (len(model.accuracy_record)) * evaluate_step
            data_list.append(iter_accuracy)
        datas.append(data_list)
    sns.heatmap(np.array(datas), xticklabels=maxlens, yticklabels=lrs)
    plt.title('Iteration when Accuracy comes to 1.0')
    plt.xlabel('MaxLens')
    plt.ylabel('LearningRate')
    plt.show()
    # datas:(#lr, #maxlen)


def experiment_hidden(device):
    """
    hidden初始化对于模型学习的影响
    """
    maxlen = 5
    lr = 0.001
    evaluate_step = 50
    legend_list = []
    repeat = 5
    loss_list = []
    accuracy_list = []
    title = 'Initial Hidden'
    initial_list = ['randn', 'randn10', 'zero', 'xavier', 'xavier_he']
    for initial in initial_list:
        loss = 0
        accuracy = 0
        for i in range(repeat):
            model = myAdvPTRNNModel(model_device=device, h_initial=initial).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
            train(1000, model, optimizer, None, evaluate_step, device, maxlen)
            loss += np.array(model.loss_record)
            accuracy += np.array(model.accuracy_record)
        loss_list.append(loss / repeat)
        accuracy_list.append(accuracy / repeat)
        legend_list.append(initial)
    visualize_list(loss_list, accuracy_list, title, legend_list, evaluate_step=evaluate_step)

def experiment_model(device):
    """
    非线性部分/LSTM/GRU对于模型学习的影响
    """
    # model_list = []
    legend_list = []
    maxlen = 5
    lr = 0.001
    evaluate_step = 50
    repeat = 5
    loss_list = []
    accuracy_list = []
    title = "Model"
    test_case = [('rnn', 'relu'), ('rnn', 'tanh'), ('lstm', ''), ('gru', '')]
    for (mode, nonlinear) in test_case:
        loss = 0
        accuracy = 0
        for i in range(repeat):
            model = myAdvPTRNNModel(model_device=device, mode=mode, nonlinear=nonlinear).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
            train(1000, model, optimizer, None, evaluate_step, device, maxlen)
            loss += np.array(model.loss_record)
            accuracy += np.array(model.accuracy_record)
        loss_list.append(loss / repeat)
        accuracy_list.append(accuracy / repeat)
        legend_list.append(mode + ':' + nonlinear)
    visualize_list(loss_list, accuracy_list, title, legend_list, evaluate_step=evaluate_step)
    # visualize(model_list, title, legend_list)

def experiment_h_not(device):
    """
    代码中是否迭代hidden_size对于模型的影响
    """
    maxlen = 5
    evaluate_step = 50
    repeat = 5
    loss_list = []
    accuracy_list = []
    legend_list = []
    title = 'Iterating H or Not- Maxlen=5'
    hs = ['True', 'False', 'Zero']
    for h in hs:
        loss = 0
        accuracy = 0
        for i in range(repeat):
            model = myAdvPTRNNModel(model_device=device, h_not=h, num_layers=1).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
            train(300, model, optimizer, None, evaluate_step, device, maxlen)
            loss += np.array(model.loss_record)
            accuracy += np.array(model.accuracy_record)
        loss_list.append(loss / repeat)
        accuracy_list.append(accuracy / repeat)
        legend_list.append(str(h))
    visualize_list(loss_list, accuracy_list, title, legend_list, evaluate_step=evaluate_step)
