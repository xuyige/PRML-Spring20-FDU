
import argparse

from handout import *


if __name__ == '__main__':
    '''
        Example: python source.py --pt=rnn --adv=rnn --epoch=1000 --batch_size=200 --train=normal --evaluate=normal
        --pt: choose rnn, lstm of gru in ptRNNModule
        --adv: choose rnn or indrnn in ptRNNModule
        --epoch: epoch of training
        --batch_size: batch size of training
        --train: choose max length of add numbers in training is 10 or 50 (normal or extreme)
        --evaluate: choose max length of add numbers in evaluate is 10 or 50 (normal or extreme)
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', choices=['rnn', 'lstm', 'gru'], default='rnn')
    parser.add_argument('--adv', choices=['rnn', 'indrnn'], default='rnn')
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--train', choices=['normal', 'extreme'], default='normal')
    parser.add_argument('--evaluate', choices=['normal', 'extreme'], default='normal')
    arg = parser.parse_args()
    pt_main(choice=arg.pt, epoch=arg.epoch, batch_size=arg.batch_size, train_set=arg.train, evaluate_set=arg.evaluate)
    pt_adv_main(choice=arg.adv, epoch=arg.epoch, batch_size=arg.batch_size, train_set=arg.train, evaluate_set=arg.evaluate)
    