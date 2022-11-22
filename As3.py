import numpy as np
import sympy as sym     #Lib for Symbolic Math
from matplotlib import pyplot
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def objective(x):
  return (x+3)**2

def derivative(x):
  return 2*(x + 3)

def gradient_descent(alpha, start, max_iter):
  x_list = list()
  x= start;
  x_list.append(x)
  for i in range(max_iter):
    gradient = derivative(x);
    x = x - (alpha*gradient);
    x_list.append(x);
  return x_list

x = sym.symbols('x')
expr = (x+3)**2.0;
grad = sym.Derivative(expr,x)
print("{}".format(grad.doit()) )
grad.doit().subs(x,2)

def gradient_descent1(expr,alpha, start, max_iter):
  x_list = list()
  x = sym.symbols('x')
  grad = sym.Derivative(expr,x).doit()  
  x_val= start;
  x_list.append(x_val)
  for i in range(max_iter):
    gradient = grad.subs(x,x_val);
    x_val = x_val - (alpha*gradient);
    x_list.append(x_val);
  return x_list

alpha = 0.1       #Step_size
start = 2         #Starting point
max_iter = 30     #Limit on iterations
x = sym.symbols('x')
expr = (x+3)**2;   #target function

X_axixx=np.array([-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
Y_axixx=objective(X_axixx)
plt.plot(X_axixx,Y_axixx,marker='o',color='b',linestyle='-');
plt.plot(X_axixx, Y_axixx, 'o', color='r');

x_cordinate = np.linspace(-15,15,100)
pyplot.plot(x_cordinate,objective(x_cordinate))
pyplot.plot(2,objective(2),'ro')

X = gradient_descent(alpha,start,max_iter)

x_cordinate = np.linspace(-5,5,100)
pyplot.plot(x_cordinate,objective(x_cordinate))

X_arr = np.array(X)
pyplot.plot(X_arr, objective(X_arr), '.-', color='red')
pyplot.show()

X= gradient_descent1(expr,alpha,start,max_iter)
X_arr = np.array(X)

x_cordinate = np.linspace(-5,5,100)
pyplot.plot(x_cordinate,objective(x_cordinate))

X_arr = np.array(X)
pyplot.plot(X_arr, objective(X_arr), '.-', color='red')
pyplot.show()


