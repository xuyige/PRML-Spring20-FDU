import random
import argparse

from handout import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', default='pt')
    parser.add_argument('--digit_len', type=int,default=100)
    parser.add_argument('--model', default='RNN')
    parser.add_argument('--lr', type=float,default=1e-3)
    parser.add_argument('--layer', type=int, default=3)
    parser.add_argument('--epoch', type=int,default=1000)
    parser.add_argument('--batch_size', type=int,default=200)
    parser.add_argument('--test_batch_size', type=int,default=200)

    arg = parser.parse_args()
    random.seed(1)
    print(type(arg),arg)
    if arg.framework == 'pt':
        pt_adv_main(arg)
    else:
        raise RuntimeError
