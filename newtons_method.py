
import math
import time

tic = time.time()

def f(x):
    d = math.exp(-(x*x))
    t = (x*x - 2*x + 2)**(5/3)
    return (d*(-6*x**5 + 12*x**4 - 10*x**3 - 8*x**2 + 12*x) + (2/15)*x - (2/15))/t

def f_prime(x):
    a = math.exp(-(x*x))
    e = (x*x - 2*x + 2)**(8/3)
    h = 12*x**8 - 48*x**7 + 82*x**6 - 24*x**5 -(386/3)*x**4 + (640/3)*x**3 - (392/3)*x**2 - 16*x +24

    return ((a*h) - .311111111*x*x + .62222222*x - .177777777)/e


g = -1
tol = 10 ** -16
defect = abs(f(g))
count = 0

while defect > tol and count < 1000:
    g = g - f(g)/f_prime(g)
    defect = abs(f(g))
    count = count + 1


toc = time.time()
runTime = toc - tic

print(g)
print(f(g))
print(count)
print(str(runTime))
