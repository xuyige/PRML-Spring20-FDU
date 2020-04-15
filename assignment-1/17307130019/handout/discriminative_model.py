import os
import sys
import math
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from handout.gaussian_distrubution import *


#计算样本均值 
#参数为nxm的矩阵，n表示样本数量，m表示维度
def get_mean(m):
	mean=np.mean(m, axis = 0)
	return mean

 
#计算类内距离
#矩阵同上
#均值表示为1xd向量，d表示维数，与样本维度相同
def intra_class(m, mean):
	nums, dimens = m.shape[:2]
	t_m = m - mean

	intra=0	
	for i in range(nums):
		x = t_m[i]
		intra += np.dot(x, x.T)
	return intra

#计算全部参数
def get_parameter(A, B, A_mean, B_mean):
    A_intra = intra_class(A, A_mean)
    B_intra = intra_class(B, B_mean)
    
    intra_all = A_intra + B_intra
    intra_all_i = intra_all.I       #矩阵求逆

    W = np.dot(intra_all_i, A_mean - B_mean)
    W_t = W.T

    gA = np.dot(A_mean, W_t)
    gB = np.dot(B_mean, W_t)
    # gA = np.dot(A_mean - 0.5 * (A_mean + B_mean), W_t)
    # gB = np.dot(B_mean - 0.5 * (A_mean + B_mean), W_t)
    # w0 = 0.5 * (gA + gB)
    return W.T, gA, gB

# Fisher Linear Discriminant 
def LDM(A, B, C, T, NA, NB, NC, label):
    A_mean = get_mean(A)
    B_mean = get_mean(B)
    C_mean = get_mean(C)

    AB = np.r_[A, B]
    AC = np.r_[A, C]
    BC = np.r_[B, C]
    print("\nDiscriminative Model")

# ------------One versus the rest --------------------------------------
    print("--------- One versus the rest-------")
    W1_t, g1A, g1BC = get_parameter(A, BC, A_mean, 0.5 * (B_mean + C_mean))
    W2_t, g2B, g2AC = get_parameter(B, AC, B_mean, 0.5 * (A_mean + C_mean))
    W3_t, g3C, g3AB = get_parameter(C, AB, C_mean, 0.5 * (A_mean + B_mean))

    rA, rB, rC, cA, cB, cC = 0, 0, 0, 0, 0, 0
    for i in range(NA + NB + NC):
        g1 = np.dot(T[i], W1_t)
        g2 = np.dot(T[i], W2_t)
        g3 = np.dot(T[i], W3_t)

        if (abs(g1 - g1A) <= abs(g1 - g1BC)) and (abs(g2 - g2B) >= abs(g2 - g2AC)) and (abs(g3 - g3C) >= abs(g3 - g3AB)):
            cA += 1
            if label[i] == 'A':
                rA += 1
        if (abs(g1 - g1A) >= abs(g1 - g1BC)) and (abs(g2 - g2B) <= abs(g2 - g2AC)) and (abs(g3 - g3C) >= abs(g3 - g3AB)):
            cB += 1
            if label[i] == 'B':
                rB += 1
        if (abs(g1 - g1A) >= abs(g1 - g1BC)) and (abs(g2 - g2B) >= abs(g2 - g2AC)) and (abs(g3 - g3C) <= abs(g3 - g3AB)):
            cC += 1
            if label[i] == 'C':
                rC += 1
    show_accuracy(rA, rB, rC, NA, NB, NC, cA, cB, cC)

# ------------One versus one -------------------------------------------
    print("--------- One versus one -----------")
    W1_t, g1A, g1B = get_parameter(A, B, A_mean, B_mean)
    W2_t, g2A, g2C = get_parameter(A, C, A_mean, C_mean)
    W3_t, g3B, g3C = get_parameter(B, C, B_mean, C_mean)

    rA, rB, rC, cA, cB, cC = 0, 0, 0, 0, 0, 0
    for i in range(NA + NB + NC):
        g1 = np.dot(T[i], W1_t)
        g2 = np.dot(T[i], W2_t)
        g3 = np.dot(T[i], W3_t)
        if (abs(g1 - g1A) <= abs(g1 - g1B)) and (abs(g2 - g2A) <= abs(g2 - g2C)):
            cA += 1
            if label[i] == 'A':
                rA += 1
        elif (abs(g1 - g1A) >= abs(g1 - g1B)) and (abs(g3 - g3B) <= abs(g3 - g3C)):
            cB += 1
            if label[i] == 'B':
                rB += 1
        elif (abs(g2 - g2A) >= abs(g2 - g2C)) and (abs(g3 - g3B) >= abs(g3 - g3C)):
            cC += 1
            if label[i] == 'C':
                rC += 1
    show_accuracy(rA, rB, rC, NA, NB, NC, cA, cB, cC)

# ------------argmax ---------------------------------------------------
    print("--------- argmax -------------------")
    W1_t, g1A, g1BC = get_parameter(A, BC, A_mean, 0.5 * (B_mean + C_mean))
    W2_t, g2B, g2AC = get_parameter(B, AC, B_mean, 0.5 * (A_mean + C_mean))
    W3_t, g3C, g3AB = get_parameter(C, AB, C_mean, 0.5 * (A_mean + B_mean))

    rA, rB, rC, cA, cB, cC = 0, 0, 0, 0, 0, 0
    for i in range(NA + NB + NC):
        g1 = np.dot(T[i] - 0.5 * (A_mean + 0.5 * (B_mean + C_mean)), W1_t)
        g2 = np.dot(T[i] - 0.5 * (B_mean + 0.5 * (A_mean + C_mean)), W2_t)
        g3 = np.dot(T[i] - 0.5 * (C_mean + 0.5 * (A_mean + B_mean)), W3_t)
        if g1 >= g2 and g1 >= g3:
            cA += 1
            if label[i] == 'A':
                rA += 1
        elif g2 >= g1 and g2 >= g3:
            cB += 1
            if label[i] == 'B':
                rB += 1
        elif g3 >= g2 and g3 >= g1:
            cC += 1
            if label[i] == 'C':
                rC += 1
    show_accuracy(rA, rB, rC, NA, NB, NC, cA, cB, cC)