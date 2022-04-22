import numpy as np

def euler_scheme(x,func,dt,*args):
    sol = x + dt * func(x,*args)
    return sol

def rk4_scheme(x,func,dt,*args):
    k1 = func(x,*args)
    k2 = func(x + dt/2*k1,*args)
    k3 = func(x + dt/2*k2,*args)
    k4 = func(x + k3*dt, *args)
    sol = x + dt/6*(k1 + 1*k2 + 2*k3 + k4)
    return sol

