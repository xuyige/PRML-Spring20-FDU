import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from handout import *


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', default='pt')
    arg = parser.parse_args()

    if arg.framework == 'pt':
        # 默认前题下运行RNN
        pt_main(device)
        pt_adv_main(device)
        # 关于学习率的实验
        experiment_lr(device)
        # 关于数字长度的实验
        experiment_maxlen(device)
        # 关于学习率和数字长度相关性的实验
        experiment_maxlen_lr(device)
        # 关于RNN层数的实验
        experiment_layer(device)
        # 关于隐藏单元的实验
        experiment_hidden(device)
        # 关于隐藏单元迭代方法和RNN层数关系的实验
        experiment_layer_h_not(device)
        # 关于隐藏单元迭代方法的实验
        experiment_h_not(device)
        # 关于不同激活函数和模型的实验
        experiment_model(device)

    elif arg.framework == 'tf':
        tf_main()
        tf_adv_main()
    else:
        raise RuntimeError
