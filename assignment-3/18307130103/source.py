import math
import numpy as np
import random

from handout import module
from handout import generate

generate.Generate(mean=[[0, 0], [-3, -0], [4, 0]], cov=[[[1, 0], [0, 1]], [[1.5, 0], [0, 1.5]], [[0.8125, -0.325], [-0.325, 0.4375]]], filename='1', n=500)
gtx, gtl = generate.Inputdata()
generate.Picture(gtx, gtl, filename='1')
gmm = module.GMM()
gmm.Train(gtx, gtl, classes=3, epoch=100, epochk=10)
label = gmm.Classify(gtx)

print(module.Checker(label, gtl))
generate.Outputdata(gtx, gtl, filename='2')
generate.Picture(gtx, gtl, filename='2')

