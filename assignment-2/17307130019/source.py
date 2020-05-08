
import argparse

from handout import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choice=['tf', 'pt'], default='pt')
    arg = parser.parse_args()

    if arg.framework == 'pt':
        pt_main()
        pt_adv_main(3000, 10, 10, 3, "RNN", 0.01, 100)
    elif arg.framework == 'tf':
        tf_main()
        tf_adv_main()
    else:
        raise RuntimeError
