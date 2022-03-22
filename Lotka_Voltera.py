from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import ipywidgets as ipw

alpha = 1
beta = 0.26
delta = 0.1
x0 = 1
y0 = 1

def pred_prey_eqns(Y, t, alpha, beta, delta):
    x,y = Y
    dx = x * (1-x) - ((alpha * x * y)/(delta + x))
    dy = beta * y * (1 - y/x)
    return np.array([dx,dy])

N = 1000
tend = 100
t = np.linspace(0,tend,N)
Y0 = [x0,y0]
sol = integrate.odeint(pred_prey_eqns, Y0, t, args = (alpha, beta, delta))
x,y = sol.T

plt.figure()
plt.grid()
plt.title('odeint')
plt.plot(t,x,'b',label='Prey')
plt.plot(t,y,'r',label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()