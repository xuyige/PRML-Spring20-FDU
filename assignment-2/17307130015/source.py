
import argparse
from handout import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['tf', 'pt'], default='pt')
    parser.add_argument('--adv', default=False, action='store_true')
    parser.add_argument('--rnn_adv', default=False, action='store_true')
    parser.add_argument('--load', default=False, action='store_true')
    parser.add_argument('--maxlen', type=int, default=250)
    parser.add_argument('--train_dataset_size', type=int, default=10000)
    arg = parser.parse_args()
    if arg.framework == 'pt':
        if arg.adv:
            pt_adv_main(arg)
        else:
            pt_main(arg)
    elif arg.framework == 'tf':
        pass
    else:
        raise RuntimeError
