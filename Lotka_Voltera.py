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
b = 1
sigma = -1
x0 = 1
y0 = 1
u10 = 1
u20 = 1

def pred_prey_eqns(Y, t, alpha, beta, delta):
    x,y = Y
    dx = x * (1-x) - ((alpha * x * y)/(delta + x))
    dy = beta * y * (1 - y/x)
    return np.array([dx,dy])

def hopf_bif(X,t,b,sigma):
    u1,u2 = X
    du1 = b*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2 = u1 + b*u2 + sigma*u2*(u1**2 + u2**2) 
    return np.array([du1,du2])


N = 1000
tend = 100
t = np.linspace(0,tend,N)
Y0 = [x0,y0]
X0 = [u10,u20]

pp_sol = integrate.odeint(pred_prey_eqns, Y0, t, args = (alpha, beta, delta))
hopf_sol = integrate.odeint(hopf_bif, X0, t, args = (b, sigma))
x,y = pp_sol.T
u1,u2 = hopf_sol.T

def odeint_plots():
    plt.figure()
    plt.grid()
    plt.title('Predator-prey equations solved with "odeint" method')
    plt.plot(t,x,'b',label='Prey')
    plt.plot(t,y,'r',label='Predator')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

    plt.figure()
    plt.grid()
    plt.title('Hopf bifurcation solved with "odeint" method')
    plt.plot(t,u1,'b',label='u_1(t)')
    plt.plot(t,u2,'r',label='u_2(t)')
    plt.xlabel('Time')
    plt.ylabel('Bifurcation value')
    plt.legend()
    plt.show()

odeint_plots()

def b_value_plots():
    beta_vals = np.arange(0.1,0.6,0.1)
    vals = np.random.random((10,len(beta_vals)))
    cols = cm.rainbow(np.linspace(0,1,vals.shape[0]))

    fig,ax = plt.subplots(2,1)

    for beta, k in zip(beta_vals, range(len(beta_vals))):
        pp_sol = integrate.odeint(pred_prey_eqns, Y0, t, args = (alpha,beta,delta))
        ax[0].plot(t, pp_sol[:,0], color = cols[k], linestyle='-', label=r"b = " + "{0:.2f}".format(beta))
        ax[1].plot(t, pp_sol[:,1], color = cols[k], linestyle='-', label=r"b = " + "{0:.2f}".format(beta))
        ax[0].legend(loc = 'best')
        ax[1].legend(loc = 'best')

    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Prey')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Predator')

    plt.show()

    b_vals = np.arange(0,1.2,0.2)
    vals = np.random.random((10,len(b_vals)))
    cols = cm.rainbow(np.linspace(0,1,vals.shape[0]))

    fig,ax = plt.subplots(2,1)

    for b, k in zip(b_vals, range(len(b_vals))):
        hopf_sol = integrate.odeint(hopf_bif, X0, t, args = (b, sigma))
        ax[0].plot(t, hopf_sol[:,0], color = cols[k], linestyle='-', label=r"$\beta$ = " + "{0:.2f}".format(b))
        ax[1].plot(t, hopf_sol[:,1], color = cols[k], linestyle='-', label=r"$\beta$ = " + "{0:.2f}".format(b))
        ax[0].legend(loc = 'best')
        ax[1].legend(loc = 'best')

    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('u_1(t)')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('u_2(t)')

    plt.show()

b_value_plots()

def phase_portrait_plots():
    plt.figure()
    prey_ics = np.linspace(1.0,5.0,17)
    for prey in prey_ics:
        Y0 = [prey, 1.0]
        Ys = integrate.odeint(pred_prey_eqns, Y0, t, args = (alpha, beta, delta))
        plt.plot(Ys[:,0], Ys[:,1], "-", label = "$x_0 =$"+str(Y0[0]))
    plt.grid()
    plt.xlabel('Prey')
    plt.ylabel('Predator')
    plt.legend(loc = 'best')
    plt.title('Prey vs Predator phase portrait')
    plt.show()

    plt.figure()
    u1_ics = np.linspace(1.0,5.0,17)
    for u1 in u1_ics:
        X0 = [u1, 1.0]
        Xs = integrate.odeint(hopf_bif, X0, t, args = (b, sigma))
        plt.plot(Xs[:,0], Xs[:,1], "-", label = "$u_0 =$"+str(X0[0]))
    plt.grid()
    plt.xlabel('u_1(t)')
    plt.ylabel('u_2(t)')
    plt.legend(loc = 'best')
    plt.title('Hopf bifurcation phase portrait')
    plt.show()

phase_portrait_plots()

def pp_euler_solver(f,Y0,t,alpha,beta,delta):
    dt = t[1] - t[0]
    lt = len(t)
    x = np.zeros([lt, len(Y0)])
    x[0] = Y0
    for k in range(lt-1):
        x[k+1] = x[k] + f(x[k], t[k], alpha, beta, delta) * dt
    return x

def hb_euler_solver(f,X0,t,b,sigma):
    dt = t[1] - t[0]
    lt = len(t)
    u1 = np.zeros([lt, len(X0)])
    u1[0] = X0
    for k in range(lt-1):
        u1[k+1] = u1[k] + f(u1[k], t[k], b, sigma) * dt
    return u1

def euler_plots():
    Ye = pp_euler_solver(pred_prey_eqns, Y0, t, alpha, beta, delta)
    Xe = hb_euler_solver(hopf_bif, X0, t, b, sigma)

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
    plt.title('Hopf bifurcation solved with Euler method')
    plt.plot(t, Xe[:,0], 'b', label = 'u_1(t)')
    plt.plot(t, Xe[:,1], 'r', label = 'u_2(t)')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Bifurcation value')
    plt.legend(loc = 'best')
    #plt.ylim([0.,2])
    plt.show()

    plt.figure()
    plt.plot(Ye[:,0], Ye[:,1], "-")
    plt.xlabel('Prey')
    plt.ylabel('Predator')
    plt.grid()
    plt.title('Phase plane of Prey vs Predator with Euler method')
    plt.show()

    plt.figure()
    plt.plot(Xe[:,0], Xe[:,1], "-")
    plt.xlabel('u_1(t)')
    plt.ylabel('u_2(t)')
    plt.grid()
    plt.title('Phase plane of Hopf bifurcation with Euler method')
    plt.show()

euler_plots()

def pp_RK4_solver(f,Y0,t,alpha,beta,delta):
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

def hb_RK4_solver(f,X0,t,b,sigma):
    dt = t[1] - t[0]
    lt = len(t)
    u1 = np.zeros([lt, len(X0)])
    u1[0] = X0
    for k in range(lt-1):
        sol1 = f(u1[k], t[k], b, sigma)
        sol2 = f(u1[k] + dt/2 * sol1, t[k] + dt/2, b, sigma)
        sol3 = f(u1[k] + dt/2 * sol2, t[k] + dt/2, b, sigma)
        sol4 = f(u1[k] + dt * sol3, t[k] + dt, b, sigma)
        u1[k+1] = u1[k] + dt/6 * (sol1+2 * sol2+2 * sol3 + sol4)
    return u1

def RK4_plots():
    Yrk4 = pp_RK4_solver(pred_prey_eqns, Y0, t, alpha, beta, delta)
    Xrk4 = hb_RK4_solver(hopf_bif, X0, t, b, sigma)


    plt.figure()
    plt.title('Predator-prey equations solved with RK4 method')
    plt.plot(t, Yrk4[:,0],'b',label='Prey')
    plt.plot(t, Yrk4[:,1],'r',label='Predator')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.ylim([0,1.2])
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.title('Hopf bifurcation solved with RK4 method')
    plt.plot(t, Xrk4[:,0],'b',label='u_1(t)')
    plt.plot(t, Xrk4[:,1],'r',label='u_2(t)')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Bifurcation value')
    #plt.ylim([0,10])
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.plot(Yrk4[:,0], Yrk4[:,1], "-")
    plt.xlabel('Prey')
    plt.ylabel('Predator')
    plt.grid()
    plt.title('Phase plane of Prey vs Predator with RK4 method')
    plt.show()

    plt.figure()
    plt.plot(Xrk4[:,0], Xrk4[:,1], "-")
    plt.xlabel('u_1(t)')
    plt.ylabel('u_2(t)')
    plt.grid()
    plt.title('Phase plane of Hopf bifurcation with RK4 method')
    plt.show()

RK4_plots()