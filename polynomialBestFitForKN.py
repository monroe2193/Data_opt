import time
import math
import matplotlib.pyplot as plt
import numpy as np

#def f(x):
#    return 1.76*x + 2.5
def f(x):
    return 2*x**4 - 0.5*x**3 + 0.5*x**2 - x + 0.25
K = 4
N = 7
a = -2
b = 2
step = (b-a)/N
X = np.linspace(a, b, N)
Y = np.zeros(N)


#A will be the container i keep the values for the coefficient matrix
A = np.zeros(2*K+1)

the_random_noise = np.random.random(N)

#for n in range(0,N):
#    X[n] = X[n] + ((-1)**n)*0.1*step*the_random_noise[n]

#fills in the Y values with the function evaluated @ each x value
for n in range(0,N):
    Y[n] = f(X[n])
#+ ((-1)**n)*0.5*the_random_noise[n]

print("xi's that we need to sum")
print(X)

#this loop fills A with the appropriate sums
for k in range(0,N+2):
    A[k] = A[k] + sum(i**k for i in X)

#this double loop fills the list 'a' with the values from A in the appropriate order
a = []
for i in range(0, K+1):
    a.append(A[i])
    for i in range(i+1, i+K+1):
        a.append(A[i])

#transform the list 'a' into an array
a = np.asarray(a)

#resize the array to the appropriate size for matrix multiplication
a = np.resize(a, (K+1, K+1))

#this loop fills the array B with the appropriate sums
B = np.zeros(K+1)
for i in range(0,K+1):
    B[i] = B[i] + sum(X**i * Y)
print("my general array for b")
print(B)



print("my attempt at solving a, B")



tic = time.time()
transformation_coefs = np.linalg.solve(a, B)
toc = time.time()
print()
print(transformation_coefs)
print(tic - toc)
print()
def g(x):
    return transformation_coefs_coefs[0] + transformation_coefs[1]*x + transformation_coefs[2]*x**2 \
        + transformation_coefs[3]*x**3 + transformation_coefs[4]*x**4
the_g_values = np.zeros(N)
error_squared = 0
for n in range(0,N):
    error_squared = error_squared + (g(X[n]) - Y[n])**2
    the_g_values[n] = g(X[n])
print()
print(math.sqrt(error_squared)/N)
plt.plot(X, Y, 'o')
plt.plot(X, the_g_values)
#plt.plot(A,C, 'r')
#plt.xlim([0,20])
#plt.ylim([0,5])
plt.show()


