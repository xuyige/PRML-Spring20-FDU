import argparse

from handout import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', choices=['tf', 'pt'], default='pt')
    arg = parser.parse_args()

    if arg.framework == 'pt':
        pt_main()
        pt_adv_main()
    elif arg.framework == 'tf':
        tf_main()
        tf_adv_main()
    else:
        raise RuntimeError
