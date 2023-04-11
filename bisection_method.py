import math
import time

a =
b =

def f(x):
    d = math.exp(-(x*x))
    e = (x*x - 2*x + 2)**(5/3)
    return (d*(-6*x**5 + 12*x**4 - 10*x**3 - 8*x**2 + 12*x) + (2/15)*x - (2/15))/e

error = (b - a) / 2
tol = 10 ** -16
c = math.nan
count = 0
tic = time.time()

if f(a) * f(b) < 0:
    while (error > tol) and (count < 1000):
        count = count + 1
        c = (a + b) / 2
        if f(a) * f(c) < 0:
            b = c
        elif f(b) * f(c) < 0:
            a = c
        error = (b - a) / 2
toc = time.time()
runTime = toc - tic

print(c)
print(f(c))
print(count)
print(runTime)

