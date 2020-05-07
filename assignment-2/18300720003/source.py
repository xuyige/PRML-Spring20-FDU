
import argparse

from handout import *


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("train_len", type=int)
   parser.add_argument("test_len", type=int)
   parser.add_argument("layer", type=int)
   parser.add_argument("rnn")
   parser.add_argument("step",type=int)
   args = parser.parse_args()
   Loss_list = []
   Accuracu_train=[]
   Accuracy_list_test = []
   pt_adv_main(args.layer,args.step,args.rnn,args.train_len,args.test_len)
   
