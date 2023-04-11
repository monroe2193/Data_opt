#import necessary packages
import numpy as np
import matplotlib.pyplot as plt

#define the function
def f(x, y):
  return np.sin(x) * np.exp(-x**2 - y**2)

#define the derivative of the function
def grad_f(x, y):
  return np.array([-2*x*np.sin(x)*np.exp(-x**2 - y**2) - 2*y*np.cos(x)*np.exp(-x**2 - y**2),
                   -2*y*np.sin(x)*np.exp(-x**2 - y**2)])

#define the step size
alpha = 0.2

#set the initial values for x and y
x_0 = 0.1
y_0 = 0.1

#plot the function
x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)

X,Y = np.meshgrid(x,y)
Z = f(X,Y)


#create an array to store the values of x and y at each step
vals = np.array([x_0, y_0])

#create a for loop to perform the gradient descent
for _ in range(50):
  #calculate the gradient at each step
  grad = grad_f(*vals)

  #update the values of x and y based on the step size and the gradient
  vals -= alpha * grad

#generate a graph of the gradient vectors
plt.quiver(X, Y, -2*X*np.sin(X)*np.exp(-X**2 - Y**2) - 2*Y*np.cos(X)*np.exp(-X**2 - Y**2),
           -2*Y*np.sin(X)*np.exp(-X**2 - Y**2), color='r', scale=10)

#generate a graph of the resulting function
plt.contour(X, Y, Z, levels=np.arange(-1,1.5,0.5))

#add a point to the graph to mark the maximum of the function
plt.scatter(*vals, c='g', marker='x', s=50)

#show the graph
plt.show()
