#######################################################
# Name: Phuc H. Lam
# NetID: plam6
# Email: plam6@u.rochester.edu
#######################################################

from EMprogram import *
from Kmeans import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Each entry of M is a d-dimensional vector center
# Each entry of COV is a (d*d) covariance matrix
# Weight_pi is a K-dimensional vector with nonnegative entries summing to 1
# FOR USERS: modify K and iter here.
K = 10
iter = 10

# Create a matrix of data
# FOR USERS: modify file name here.
with open('emData/Z') as f1:
    Lines = f1.readlines()
X = []
for line in Lines:
    lst = [float(n) for n in line.split(' ') if n.strip()]
    X.append(lst)
X = np.array(X)
N, d = X.shape

M, COV, Loss, BIC = BIC_score(X, K, iter)
print(BIC)

# Plot loss function
plt.figure()
X_track = np.array(range(iter))
plt.plot(X_track, Loss, color="red")
plt.show()

# Plot datasets (for dataset A, B, C)
# Plot Gaussian clusters
"""
X_ = np.transpose(X)
x = X_[0]
y = X_[1]
plt.figure()
plt.scatter(x, y, s=0.1)
plt.show()
"""

# Write the mean, covariance, and weights into file
# FOR USERS: modify file name here
"""
f2 = open('ResultParam/Z.txt', 'a+')

f2.write('Number of clusters: ' + str(K) + '\n')
f2.write('Number of iterations: ' + str(iter) + '\n')

f2.write('Mean: \n')
for i in range(K):
    np.savetxt(f2, M[i], delimiter=' ')
    f2.write('\n')
f2.write('\n')

f2.write('Covariance: \n')
for i in range(K):
    np.savetxt(f2, COV[i], delimiter=' ')
    f2.write('\n')
f2.write('\n')

f2.write('Weights: \n')
np.savetxt(f2, Weight_pi, delimiter=' ')
f2.write('\n')

f2.write('Train loss: \n')
np.savetxt(f2, Train_loss, delimiter=' ')
f2.write('\n')

f2.write('Test loss: \n')
np.savetxt(f2, Test_loss, delimiter=' ')
f2.write('\n')

f2.write('****************************************')
f2.write('\n')
f2.close()
"""

# Plot train and test loss
"""
plt.figure()
Xloss = np.array(range(iter))
Test_loss_check_change = [0]
for i in range(1, iter):
    if (Train_loss[i] > Train_loss[i-1]): Test_loss_check_change.append(100)
    elif (Train_loss[i] < Train_loss[i-1]): Test_loss_check_change.append(-100)
    else: Test_loss_check_change.append(0)
Test_loss_check_change = np.array(Test_loss_check_change)
plt.plot(Xloss, Train_loss, color="red")
plt.plot(Xloss, Test_loss, color="blue")
plt.plot(Xloss, Test_loss_check_change, color="green")
plt.show()
"""