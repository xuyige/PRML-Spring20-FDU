from matplotlib import pyplot as plt
import numpy as np
import random
import math

'''
##################################################### Create Data ######################################################
mean1 = np.array([3,3])
mean2 = np.array([0,0])
mean3 = np.array([-3,3])
cov = ([1,0],[0,1])

dot_num = 100
f1xy = np.random.multivariate_normal(mean1,cov,dot_num)
f2xy = np.random.multivariate_normal(mean2,cov,dot_num)
f3xy = np.random.multivariate_normal(mean3,cov,dot_num)

file = open("data.data",'wb')
i1 = 0
i2 = 0
i3 = 0
i = 0

while(i<240):
    r = random.randint(0,2)
    if r == 0 and i1 < 80:
        file.write(str.encode(str(np.float(f1xy[i1,0]))))
        file.write(b'\n')
        file.write(str.encode(str(np.float(f1xy[i1,1]))))
        file.write(b'\n')
        file.write(b'0')
        file.write(b'\n')
        i1 += 1
        i = i + 1
    elif r == 1 and i2 < 80:
        file.write(str.encode(str(np.float(f2xy[i2, 0]))))
        file.write(b'\n')
        file.write(str.encode(str(np.float(f2xy[i2, 1]))))
        file.write(b'\n')
        file.write(b'1')
        file.write(b'\n')
        i2 += 1
        i = i + 1
    elif r == 2 and i3 < 80:
        file.write(str.encode(str(np.float(f3xy[i3, 0]))))
        file.write(b'\n')
        file.write(str.encode(str(np.float(f3xy[i3, 1]))))
        file.write(b'\n')
        file.write(b'2')
        file.write(b'\n')
        i3 += 1
        i = i + 1

i1 = 0
i2 = 0
i3 = 0
i = 0
while(i<60):
    r = random.randint(0,2)
    if r == 0 and i1 < 20:
        file.write(str.encode(str(np.float(f1xy[i1,0]))))
        file.write(b'\n')
        file.write(str.encode(str(np.float(f1xy[i1,1]))))
        file.write(b'\n')
        file.write(b'0')
        file.write(b'\n')
        i1 += 1
        i = i + 1
    elif r == 1 and i2 < 20:
        file.write(str.encode(str(np.float(f2xy[i2, 0]))))
        file.write(b'\n')
        file.write(str.encode(str(np.float(f2xy[i2, 1]))))
        file.write(b'\n')
        file.write(b'1')
        file.write(b'\n')
        i2 += 1
        i = i + 1
    elif r == 2 and i3 < 20:
        file.write(str.encode(str(np.float(f3xy[i3, 0]))))
        file.write(b'\n')
        file.write(str.encode(str(np.float(f3xy[i3, 1]))))
        file.write(b'\n')
        file.write(b'2')
        file.write(b'\n')
        i3 += 1
        i = i + 1

file.close()
'''
############################################## Load Data ###############################################################
file = open("data.data",'rb')
size = 0
sample_list = [ [0] * 3 for i in range(3000)]
while(1):
    preread = file.readline()
    if(preread == b''):
        break
    sample_list[size][0] = float(preread)
    sample_list[size][1] = float(file.readline())
    sample_list[size][2] = float(file.readline())
    size = size + 1

############################################## Discriminative Model ####################################################

def softmax(W,x,b):
    y = W * x + b
    total = pow(math.e,y[0,0]) + pow(math.e, y[1,0]) + pow(math.e, y[2,0])
    x1 = pow(math.e, y[0,0]) / total
    x2 = pow(math.e, y[1,0]) / total
    x3 = pow(math.e, y[2,0]) / total
    result = np.mat([[x1, x2, x3]])
    return result

def K(k):
    vectorK = np.mat([[0,0,0]])
    k = int(k)
    vectorK[0,k] = 1
    return vectorK


W = np.mat([[1,1],[1,1],[1,1]])
b = np.mat([[1,1,1]]).T
alpha = 0.008   # Learning Rate #
j = 0
i = 0
while(j<10000):
    if(i == 4*size/5):
        i = 0
    x = np.mat([[sample_list[i][0],sample_list[i][1]]]).T
    p = softmax(W,x,b).T
    k = K(sample_list[i][2]).T

    gb = p - k
    gw = gb * x.T
    W = W - alpha * gw
    b = b - alpha * gb
    i += 1
    j += 1
                            ### Test ###
i = int(4 * size / 5)
counter = 0
while(i < size):
    x = np.mat([[sample_list[i][0],sample_list[i][1]]]).T
    p = softmax(W,x,b)
    p = np.array(p)
    index = np.argmax(p)
    if(index == sample_list[i][2]):
        counter += 1
    i += 1

print(counter,'/',size / 5)

############################################## Generative Model ########################################################

miuA = np.mat([[0,0]]).T
miuB = np.mat([[0,0]]).T
miuC = np.mat([[0,0]]).T

i = 0
while(i < 4 * size / 5):
    x = np.mat([[sample_list[i][0], sample_list[i][1]]]).T
    if sample_list[i][2] == 0:
        miuA = miuA + x
    elif sample_list[i][2] == 1:
        miuB = miuB + x
    elif sample_list[i][2] == 2:
        miuC = miuC + x
    i += 1

miuA /= (4 * size / 15)
miuB /= (4 * size / 15)
miuC /= (4 * size / 15)

i = 0
CorrA = np.mat([[0,0],[0,0]])
CorrB = np.mat([[0,0],[0,0]])
CorrC = np.mat([[0,0],[0,0]])


while(i < 4 * size / 5):
    x = np.mat([[sample_list[i][0], sample_list[i][1]]]).T
    if sample_list[i][2] == 0:
        CorrA = CorrA + (x - miuA) * (x - miuA).T
    elif sample_list[i][2] == 1:
        CorrB = CorrB + (x - miuB) * (x - miuB).T
    elif sample_list[i][2] == 2:
        CorrC = CorrC + (x - miuC) * (x - miuC).T
    i += 1

CorrA /= (4 * size / 15)
CorrB /= (4 * size / 15)
CorrC /= (4 * size / 15)

                                    ### Test ###
i = int(4 * size / 5)
counter = 0
while(i < size):
    x = np.mat([[sample_list[i][0], sample_list[i][1]]]).T
    PA = (1 / 2 * math.pi * pow(np.linalg.det(CorrA) , 0.5 )) * pow(math.e , (-0.5 * (x - miuA).T * np.linalg.inv(CorrA) * (x - miuA))[0,0])
    PB = (1 / 2 * math.pi * pow(np.linalg.det(CorrB) , 0.5 )) * pow(math.e , (-0.5 * (x - miuB).T * np.linalg.inv(CorrB) * (x - miuB))[0,0])
    PC = (1 / 2 * math.pi * pow(np.linalg.det(CorrC) , 0.5 )) * pow(math.e , (-0.5 * (x - miuC).T * np.linalg.inv(CorrC) * (x - miuC))[0,0])
    vali = np.mat([[PA,PB,PC]])
    index = np.argmax(vali)
    if (index == sample_list[i][2]):
        counter += 1
    i += 1
print(counter,'/',size/5)
