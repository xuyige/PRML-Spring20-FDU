
import argparse
# import fitlog

from handout import *

######hyper
dim = 32
layers = 2
epochs = 500
lr = 0.001
model = "LSTM"
######hyper


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--framework', default='pt')
    arg = parser.parse_args()
    # fitlog.commit(__file__)             
    # fitlog.add_hyper_in_file(__file__)  

    if arg.framework == 'pt':
        print('basic model')
        pt_main(dim, layers, epochs, lr)
        print('advanced model')
        pt_adv_main(dim, layers, epochs, lr, model)
    elif arg.framework == 'tf':
        pass
        # tf_main()
        # tf_adv_main()
    else:
        raise RuntimeError

    # fitlog.finish()
