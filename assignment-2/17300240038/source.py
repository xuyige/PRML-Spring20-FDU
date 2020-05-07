import argparse

from handout import *
from handout.logger import Logger 

C = argparse.ArgumentParser()
C.add_argument("--d" , type = int , default = 32)
C.add_argument("--max_length" , type = float , default = 100)
C.add_argument("--batch_size" , type = float , default = 32)
C.add_argument("--eval_size" , type = float , default = 2000)
C = C.parse_args()

C.logger = Logger(["write"] , log_path = "log.log")
C.R 	 = Logger(["write" , "print"] , log_path = "report.log")

if __name__ == '__main__':

	C.R.log("------- base rnn model -------")
	pt_main(C)
	C.R.log("------- adv  rnn model -------")
	pt_adv_main(C)
