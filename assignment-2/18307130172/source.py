import random
import argparse

from handout import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['tf', 'pt'], default='pt')
    parser.add_argument('--len', type=int, default=10)
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--relabel', action="store_true")
    arg = parser.parse_args()

    if arg.framework == 'pt':
        mp = list(range(10))
        if arg.relabel:
            random.shuffle(mp)
        # print(mp)

        pt_main(arg.len, mp, arg.cuda)
        pt_adv_main(arg.len, arg.cuda)
    elif arg.framework == 'tf':
        tf_main()
        tf_adv_main()
    else:
        raise RuntimeError
