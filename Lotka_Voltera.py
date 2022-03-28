import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import ipywidgets as ipw
import random
import matplotlib.cm as cm

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

beta_vals = np.arange(0.1,0.6,0.1)
vals = np.random.random((10,len(beta_vals)))
cols = cm.rainbow(np.linspace(0,1,vals.shape[0]))

fig,ax = plt.subplots(2,1)

for beta, k in zip(beta_vals, range(len(beta_vals))):
    sol = integrate.odeint(pred_prey_eqns, Y0, t, args = (alpha,beta,delta))
    ax[0].plot(t, sol[:,0], color = cols[k], linestyle='-', label=r"b = " + "{0:.2f}".format(beta))
    ax[1].plot(t, sol[:,1], color = cols[k], linestyle='-', label=r"b = " + "{0:.2f}".format(beta))
    ax[0].legend(loc = 'best')
    ax[1].legend(loc = 'best')

ax[0].grid()
ax[1].grid()
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Prey')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Predator')

plt.show()

plt.figure()
prey_ics = np.linspace(1.0,5.0,17)
for prey in prey_ics:
    Y0 = [prey, 1.0]
    Ys = integrate.odeint(pred_prey_eqns, Y0, t, args = (alpha, beta, delta))
    plt.plot(Ys[:,0], Ys[:,1], "-", label = "$x_0 =$"+str(Y0[0]))
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.legend(loc = 'best')
plt.title('Prey vs Predator phase portrait')
plt.show()

def euler_solver(f,Y0,t,alpha,beta,delta):
    dt = t[1] - t[0]
    lt = len(t)
    x = np.zeros([lt, len(Y0)])
    x[0] = Y0
    for k in range(lt-1):
        x[k+1] = x[k] + f(x[k], t[k], alpha, beta, delta) * dt
    return x

Ye = euler_solver(pred_prey_eqns, Y0, t, alpha, beta, delta)
plt.figure()
plt.title('Predator-prey equations solved with Euler method')
plt.plot(t, Ye[:,0], 'b', label = 'Prey')
plt.plot(t, Ye[:,1], 'r', label = 'Predator')
plt.grid()
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend(loc = 'best')
plt.ylim([0.,1.2])
plt.show()

plt.figure()
plt.plot(Ye[:,0], Ye[:,1], "-")
plt.xlabel('Prey')
plt.ylabel('Predator')
plt.grid()
plt.title('Phase plane of Prey vs Predator with Euler method')
plt.show()

def RK4_solver(f,Y0,t,alpha,beta,delta):
    dt = t[1] - t[0]
    lt = len(t)
    x = np.zeros([lt, len(Y0)])
    x[0] = Y0
    for k in range(lt-1):
        sol1 = f(x[k], t[k], alpha, beta, delta)
        sol2 = f(x[k] + dt/2 * sol1, t[k] + dt/2, alpha, beta, delta)
        sol3 = f(x[k] + dt/2 * sol2, t[k] + dt/2, alpha, beta, delta)
        sol4 = f(x[k] + dt * sol3, t[k] + dt, alpha, beta, delta)
        x[k+1] = x[k] + dt/6 * (sol1+2 * sol2+2 * sol3 + sol4)
    return x