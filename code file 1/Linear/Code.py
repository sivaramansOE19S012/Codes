"""
Created on Sat Dec  5 23:15:46 2020

@author: Sivaraman Sivaraj, Suresh Rajendran
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

data  = pd.read_csv('D.csv', header = 0)

_X = data.iloc[:,0].tolist()
_Y = data.iloc[:,1].tolist()

X = np.array(_X)
Y = np.array(_Y)
XY = X*Y
XX = X*X

def slope(x,y,xx,xy):
    s_x = np.sum(x)
    s_y = np.sum(y)
    s_xy = np.sum(xy)
    s_xx = np.sum(xx)
    N = len(x)
    num = (N*s_xy) - (s_x * s_y)
    den = (N*s_xx) - (s_x**2)
    m = num/den
    return m


def intercept(m,x,y):
    s_x = np.sum(x)
    s_y = np.sum(y)
    N = len(x.tolist())
    num = s_y - (m*s_x)
    den = N
    b = num/den
    return b


m = slope(X,Y,XX,XY) #slope calculation by least square method
c = intercept(m, X, Y) #intercept calcultion

print("The Value of slope:", m)
print("The Value of intercept",c)


def predict(m,c,x):
    xt = x.tolist()
    Y_predicted = list()
    for i in range(len(xt)):
        temp = (m*xt[i])+c
        Y_predicted.append(temp)
    return np.array(Y_predicted)


Y_p = predict(m,c,X) # predicted value

def error(y,y_p):
    a = np.array(y.tolist())
    b = np.array(y_p.tolist())
    c = a-b
    return c.tolist()
    
Error = error(Y,Y_p) # error


def MSE(y,y_p):
    a = np.array(y.tolist())
    b = np.array(y_p.tolist())
    c = a-b
    mse = c**2
    return mse

Mean_square_error = MSE(Y,Y_p) # for pltting


def error_plot(y,y_p):
    plt.figure(figsize=(9,6))
    plt.plot(y,'g', label = "Original Value")
    plt.plot(y_p,'r', label = "Predicted Value")
    plt.title("Linear Regression - Least Square Method")
    plt.xlabel("sample point - X value")
    plt.ylabel("Function value - Y value")
    plt.legend(loc = "best")
    plt.grid()
    plt.show()
    
# error_plot(Y,Y_p)


def MSE_plot(mse):
    plt.figure(figsize=(9,6))
    plt.plot(mse,'g',label="Mean Square Error")
    plt.ylabel("MSE - Scalar Value")
    plt.xlabel("Sample Points - X values")
    plt.legend(loc = "best")
    plt.grid()
    plt.show()
    
# MSE_plot(Mean_square_error)





"""
#######################################
#### Genetic Algorithm ################
#######################################

"""


    



    
    

