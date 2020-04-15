import os
import csv
os.sys.path.append('..')
# from handout import *
import math
import numpy as np
import matplotlib.pyplot as plt

"""
A_u = -3
A_sig = math.sqrt(0.2)
B_u = 0
B_sig = math.sqrt(1)
C_u = 1
C_sig = math.sqrt(4)
A_x = np.linspace(A_u - 3 * A_sig, A_u + 3 * A_sig, N)
B_x = np.linspace(B_u - 3 * B_sig, B_u + 3 * B_sig, N)
C_x = np.linspace(C_u - 3 * C_sig, C_u + 3 * C_sig, N)
A_y = np.exp(-(A_x - A_u) ** 2 / (2 * A_sig ** 2)) / (math.sqrt(2 * math.pi) * A_sig)
B_y = np.exp(-(B_x - B_u) ** 2 / (2 * B_sig ** 2)) / (math.sqrt(2 * math.pi) * B_sig)
C_y = np.exp(-(C_x - C_u) ** 2 / (2 * C_sig ** 2)) / (math.sqrt(2 * math.pi) * C_sig)

plt.title("Gaussian") 

plt.xlabel("x")
plt.ylabel("y")
plt.plot(A_x, A_y, 'ro')
plt.plot(B_x, B_y, 'go')
plt.plot(C_x, C_y, 'bo')
plt.show()

# for i in range(N):
"""
import numpy as np
def generate_gaussian_data(NA, NB, NC):
    A_mean = (-1, -1)
    A_cov = [[0.5, 0], [0, 0.5]]
    A = np.random.multivariate_normal(A_mean, A_cov, size = NA)

    B_mean = (2, 2)
    B_cov = [[2, 0.2], [0.2, 0.5]]
    B = np.random.multivariate_normal(B_mean, B_cov, size = NB)

    C_mean = (-2, 3)
    C_cov = [[0.5, 0], [0, 0.5]]
    C = np.random.multivariate_normal(C_mean, C_cov, size = NC)

    with open('dataset.data', 'w') as f:
        f.write(str(NA + NB + NC) + "\n")
        for i in range(NA):
            f.write("A " + str(A[i][0]) + " " + str(A[i][1]) + "\n")
        for i in range(NB):
            f.write("B " + str(B[i][0]) + " " + str(B[i][1]) + "\n")
        for i in range(NC):
            f.write("C " + str(C[i][0]) + " " + str(C[i][1]) + "\n")
    plt.plot(A[:, 0], A[:, 1], 'ro')
    plt.plot(B[:, 0], B[:, 1], 'bo')
    plt.plot(C[:, 0], C[:, 1], 'go')
    plt.show()
'''
with open('dataset.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerows(A)
    f_csv.writerows(B)
    f_csv.writerows(C)
'''

def process_gaussian_data():
    dic = {'A': [], 'B': [], 'C': []}
    label = []
    with open('dataset.data', 'r') as f:
        line = f.readline()
        N = int(line)
        for i in range(N):
            t = f.readline()
            t = t.split()
            dic[t[0]].append([float(t[1]), float(t[2])])
    NA, NB, NC = len(dic['A']), len(dic['B']), len(dic['C'])
    N_test = int(0.2 * (NA + NB + NC) / 3)
    A = np.mat(dic['A'][: int(NA * 0.8)])
    B = np.mat(dic['B'][: int(NB * 0.8)])
    C = np.mat(dic['C'][: int(NC * 0.8)])
    # print(A, B, C)
    T = np.concatenate((dic['A'][int(NA * 0.8):], dic['B'][int(NB * 0.8):], dic['C'][int(NC * 0.8):]), axis=0)
    label = np.concatenate((np.array(['A'] * (NA - int(NA * 0.8))), np.array(['B'] * (NB - int(NB * 0.8))), np.array(['C'] * (NC - int(NC * 0.8)))), axis=0)
    # print(label)
    return A, B, C, T, NA - int(NA * 0.8), NB - int(NB * 0.8), NC - int(NC * 0.8), label


def show_accuracy(rA, rB, rC, NA, NB, NC, cA, cB, cC):
    print("Accuracy of A class: ", rA / NA)
    print("Accuracy of B class: ", rB / NB)
    print("Accuracy of C class: ", rC / NC)
    print("Accuracy of all class: ", (rA + rB + rC) / (NA + NB + NC))
    print("Expected A B C:\t", NA, NB, NC)
    print("Class A B C: \t", cA, cB, cC, (NA + NB + NC) - cA - cB - cC)