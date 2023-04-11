
import time
import math
import random
import matplotlib.pyplot as plt
import numpy as np
def s(x):
    return  1/(1 + np.exp(-x))
def sp(x):
    return np.exp(-x)/(1+np.exp(-x))**2
X = np.loadtxt("dataTrainingDataSet1_X.txt", dtype=float)
Y = np.loadtxt("dataTrainingDataSet1_Y.txt", dtype=float)
N = X.size

#N neurons:
#Weights and bias:
B =  2*np.random.rand(5) - 1
W =  2*np.random.rand(8) - 1
B0 = np.copy(B)
W0 = np.copy(W)
def neuralNetwork(x):
    sum = 0
    for n in range(4):
        sum = sum + W[4+n]*s(W[n]*x + B[n])
    return s(sum+B[4])
def neuralGradient(W,B):
    grad = np.zeros(13)
    for n in range(N):
        xn = X[n]
        zn = W[4]*s(W[0]*xn+B[0]) + \
             W[5]*s(W[1]*xn+B[1]) + \
             W[6]*s(W[2]*xn+B[2]) + \
             W[7]*s(W[3]*xn+B[3]) + B[4]
        E1 = sp(zn)*W[4]*sp(W[0]*xn+B[0])*xn
        E2 = sp(zn)*W[5]*sp(W[1]*xn+B[1])*xn
        E3 = sp(zn)*W[6]*sp(W[2]*xn+B[2])*xn
        E4 = sp(zn)*W[7]*sp(W[3]*xn+B[3])*xn
        E5 = sp(zn)*s(W[0]*xn+B[0])
        E6 = sp(zn)*s(W[1]*xn+B[1])
        E7 = sp(zn)*s(W[2]*xn+B[2])
        E8 = sp(zn)*s(W[3]*xn+B[3])
        E9 = sp(zn)*W[4]*sp(W[0]*xn+B[0])
        E10 = sp(zn)*W[5]*sp(W[1]*xn+B[1])
        E11 = sp(zn)*W[6]*sp(W[2]*xn+B[2])
        E12 = sp(zn)*W[7]*sp(W[3]*xn+B[3])
        E13 = sp(zn)
        grad[0] = grad[0] + 2*(neuralNetwork(X[n]) - Y[n])*E1 
        grad[1] = grad[1] + 2*(neuralNetwork(X[n]) - Y[n])*E2
        grad[2] = grad[2] + 2*(neuralNetwork(X[n]) - Y[n])*E3
        grad[3] = grad[3] + 2*(neuralNetwork(X[n]) - Y[n])*E4
        grad[4] = grad[4] + 2*(neuralNetwork(X[n]) - Y[n])*E5
        grad[5] = grad[5] + 2*(neuralNetwork(X[n]) - Y[n])*E6   
        grad[6] = grad[6] + 2*(neuralNetwork(X[n]) - Y[n])*E7 
        grad[7] = grad[7] + 2*(neuralNetwork(X[n]) - Y[n])*E8
        grad[8] = grad[8] + 2*(neuralNetwork(X[n]) - Y[n])*E9
        grad[9] = grad[9] + 2*(neuralNetwork(X[n]) - Y[n])*E10
        grad[10] = grad[10] + 2*(neuralNetwork(X[n]) - Y[n])*E11
        grad[11] = grad[11] + 2*(neuralNetwork(X[n]) - Y[n])*E12  
        grad[12] = grad[12] + 2*(neuralNetwork(X[n]) - Y[n])*E13
    return grad
def costFunction(X,Y):
    sum = 0
    for n in range(N):
        sum = sum + (neuralNetwork(X[n]) - Y[n])**2
    return sum
#Train the Network:
parmVector = np.zeros(13)
parmVector[0:7] = W[0:7]
parmVector[8:12] = B[0:4] 
learningRate = 0.25
K = 1000
costs = np.zeros(K)
grads = np.zeros(K)
for k in range(K):
    eta = neuralGradient(W,B)
    parmVector = parmVector - learningRate*eta
    W[0:7] = parmVector[0:7]
    B[0:4] = parmVector[8:12]
    costs[k] = costFunction(X,Y)
    grads[k] = np.linalg.norm(eta)
print()  
print(costFunction(X,Y))
print(np.linalg.norm(eta))
print()
eta = neuralGradient(W,B)
numPoints = 1000
X_axis = np.linspace(-8, 8, numPoints)
nX = np.zeros(numPoints)
for n in range(numPoints):
    nX[n] =  neuralNetwork(X_axis[n])
    


