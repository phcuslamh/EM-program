#######################################################
# Name: Phuc H. Lam
# NetID: plam6
# Email: plam6@u.rochester.edu
#######################################################

from more_itertools import partition
import numpy as np
import random as rand
import random as rand
from scipy.stats import multivariate_normal 
from Kmeans import *

# Divide into training/test set 
# Roughly 1/3 of the data is in the test set
def partition_data(data):
    Train = []
    Test = []
    for i in range(len(data)):
        if (rand.randint(0, 2) % 3 == 0):
            Test.append(data[i])
        else:
            Train.append(data[i])
    Train = np.array(Train)
    Test = np.array(Test)
    return Train, Test

# Compute multivariate normal pdf
def normal_pdf(X1, M1, COV1):
    n = len(X1)
    M2 = np.reshape(M1, (n, ))
    X2 = np.reshape(X1, (n, ))
    var = multivariate_normal(mean=M2, cov=COV1)
    return var.pdf(X2)

# Compute log_likelihood ln p(X | PI, M, COV) according to (9.14), p. 433
def compute_log_like(X1, M1, COV1, PI1):
    S = 0
    N1 = X1.shape[0]
    K1 = M1.shape[0]
    for n in range(N1):
        S_comp = 0
        x = X1[n]
        mat = np.zeros(K1)
        for k in range(K1):
            mat[k] += normal_pdf(x, M1[k], COV1[k])
        S_comp = np.average(mat, axis=0, weights=PI1)
        S_comp = np.log(S_comp)
        S += S_comp
    return S

# E step 
# Input: X1 is an N*d matrix, M1 is a K*d matrix, COV1 is a K*d*d matrix, PI1 is a 1*K matrix
# Return: a N*K matrix with \gamma(z_{n, k}) being the (n, k)-entry
def E_step(X1, M1, COV1, PI1):
    N1 = X1.shape[0]
    K1 = M1.shape[0]
    Resp = np.zeros((N1, K1))
    for n in range(N1):
        for k in range(K1):
            Resp[n][k] += PI1[k] * normal_pdf(X1[n], M1[k], COV1[k])
        if (np.sum(Resp[n]) > 0):
            Resp[n] = Resp[n] / np.sum(Resp[n])
    return Resp 

# M step
# Input: X1 is an N*d matrix, Resp is a N*K matrix
# Return: (M, COV, Pi) new
def M_step(X1, Resp):
    N1, d1 = X1.shape
    K1 = Resp.shape[1]
    M1 = np.zeros((K1, d1))
    COV1 = np.zeros((K1, d1, d1))
    PI1 = np.zeros(K1)
    for k in range(K1):
        N_k = np.sum(Resp[:, k])
        PI1[k] += N_k / N1

        for n in range(N1):
            M1[k] += Resp[n][k] * X1[n]
        if (N_k > 0):
            M1[k] = M1[k] / N_k

        for n in range(N1):
            a = X1[n] - M1[k]
            aT = np.transpose(a) 
            COV1[k] += Resp[n][k] * np.matmul(aT, a)
        if (N_k > 0):
            COV1[k] = COV1[k] / N_k
        COV1[k] += 1e-4 * np.eye(d1)
    M1 = np.array(M1)
    COV1 = np.array(COV1)
    PI1 = np.array(PI1)
    return M1, COV1, PI1

# Calculate BIC score
def BIC_score(X1, K1, iter):
    N1, d1 = X1.shape

    # Initialize the parameters
    # M_copy = 2*np.random.random((K1, d1)) - 1
    M_copy = Kmeans_algo(X1, 100, K1)
    COV_copy = np.random.random((K1, d1, d1))
    for i in range(K1):
        # COV_copy[i] = COV_copy[i].dot(COV_copy[i].transpose())
        COV_copy[i] = np.eye(d1)
    PI_copy = np.ones(K1) / K1

    # Train the parameters and calculate the loss
    Loss = []
    for i in range(iter):
        Respo1 = E_step(X1, M_copy, COV_copy, PI_copy)
        M_copy, COV_copy, PI_copy = M_step(X1, Respo1)
        Loss.append(compute_log_like(X1, M_copy, COV_copy, PI_copy))
        print("Iteration ", i+1, " completed.")
    Loss = np.array(Loss)

    # Calculate BIC score
    num_para = K1*d1*(d1+3) / 2 + K1 - 1 
    BIC = num_para * np.log(N1) - 2 * np.amax(Loss)
    return M_copy, COV_copy, Loss, BIC