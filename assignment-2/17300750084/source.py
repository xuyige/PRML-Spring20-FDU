import argparse
from handout import *
from handout.pt import pt_main, pt_adv_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['tf', 'pt'], default='pt')
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--type', choices=['gru', 'rnn', 'lstm'], default='lstm')
    parser.add_argument('--len', type=int, default=50)
    arg = parser.parse_args()

    if arg.framework == 'pt':
        pt_main()
        pt_adv_main(arg)
    elif arg.framework == 'tf':
        tf_main()
        tf_adv_main()
    else:
        raise RuntimeError
