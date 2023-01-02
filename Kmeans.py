#######################################################
# Name: Phuc H. Lam
# NetID: plam6
# Email: plam6@u.rochester.edu
#######################################################

import numpy as np
import random as rand

# Update the assignment of the data points to each centers
# Input: X1 is an N*d matrix, M1 is a K*d matrix
# Output: a 1*N matrix with entries in {1, ..., K} (assignment to centers)
def update_assign(X1, M1):
    Cent = []
    for n in range(len(X1)):
        c = 0
        curr_dist = np.linalg.norm(X1[n] - M1[0])
        for k in range(1, len(M1)):
            if (np.linalg.norm(X1[n] - M1[k]) < curr_dist):
                curr_dist = np.linalg.norm(X1[n] - M1[k])
                c = k
        Cent.append(c)
    Cent = np.array(Cent)
    return Cent

# Update the centers
# Input: X1 is an N*d matrix, K is the number of clusters, Cent is the assignment to centers with size 1*N
# Output: a K*d matrix, where each row is a center
def update_cent(X1, K, Cent):
    M1 = []
    for k in range(K):
        d = X1.shape[1]
        new_cent = np.zeros((1, d))
        num = 0
        for i in range(len(Cent)):
            if (Cent[i] == k):
                new_cent += X1[i]
                num += 1
        if (num > 0): 
            new_cent = new_cent / num
        M1.append(new_cent)
    M1 = np.array(M1)
    return M1

# Run K-means algorithm
# Input: X1 is an N*d matrix, iter is the number of iterations, K is the number of clusters
# Output: a K*d matrix of K centers
def Kmeans_algo(X1, iter, K):
    M1 = X1[np.random.randint(X1.shape[0], size=K), :]
    for i in range(iter):
        Assignm = update_assign(X1, M1)
        M1 = update_cent(X1, K, Assignm)
    print("K-means completed.")
    return M1