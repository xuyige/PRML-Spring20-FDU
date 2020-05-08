
import argparse
from handout import *


if __name__ == '__main__':
    # '''
    # Example:
    #
    # python source.py --algo=rnn --num=1 --hid=64 --dim=64 --step=1000 --trl=6 --tes=1000
    #
    # '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['tf', 'pt'], default='pt')
    parser.add_argument('--algo', choices=['rnn', 'lstm', 'gru'], default='rnn')
    parser.add_argument('--num', type=int, default=2)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--step', type=int, default=500)
    parser.add_argument('--trl', type=int, default=10)
    parser.add_argument('--tes', type=int, default=100)
    arg = parser.parse_args()

    if arg.algo == 'rnn':
        MOD = 'RNN'
    elif arg.algo == 'lstm':
        MOD = 'LSTM'
    else:
        MOD = 'GRU'

    if arg.framework == 'pt':
        pt_main()
        pt_adv_main(model_name=MOD, num_layers=arg.num, hidden_size=arg.hid, dim=arg.dim, steps=arg.step, train_length=arg.trl, sequence_length=arg.tes)
    # elif arg.framework == 'tf':
    #     tf_main()
    #     tf_adv_main()
    # else:
    #     raise RuntimeError
