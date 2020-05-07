"""
Environment:
    Tensorflow 2.0.0a
"""
import argparse
import logging
import time

from handout import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['normal', 'advanced', 'cascade'], default='normal')
parser.add_argument('--optimizer', choices=['Adam', 'SGD', 'RMSprop'], default='Adam')
parser.add_argument('--max_length', type=int, default=20)
parser.add_argument('--steps', type=int, default=3000)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--evaluate_batch_size', type=int, default=2001)
parser.add_argument('--plot', type=bool, default=True)
parser.add_argument('--unit_max_length', type=int, default=5)
parser = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('')
handler.setFormatter(formatter)
logger.addHandler(handler)

parser.logger = logger

if __name__ == '__main__':
    parser.logger.info('RNN model simulating an adder:\n' +
                       ''.join(['-'] * 30) + '\n'
                                             'Model: ' + parser.model + '\n' +
                       'Max length: ' + str(parser.max_length) + '\n' +
                       'Optimizer: ' + parser.optimizer + '\n' +
                       'Learning rate: ' + str(parser.learning_rate) + '\n' +
                       'Steps: ' + str(parser.steps) + '\n' +
                       'Batch size: ' + str(parser.batch_size) + '\n' +
                       'Evaluate batch size: ' + str(parser.evaluate_batch_size))

    if parser.model == 'normal':
        tf_main(parser)
    elif parser.model == 'advanced':
        tf_adv_main(parser)
    elif parser.model == 'cascade':
        tf_cascade_main(parser)
    else:
        raise RuntimeError
