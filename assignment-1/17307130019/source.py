import os
import sys
import math
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from handout.gaussian_distrubution import *
from handout.discriminative_model import *
from handout.generative_model import *


if __name__=='__main__':
    generate_gaussian_data(150, 150, 150)
    A, B, C, T, NA, NB, NC, label= process_gaussian_data()
    LDM(A, B, C, T, NA, NB, NC, label)
    LGM(A, B, C, T, NA, NB, NC, label)

    
