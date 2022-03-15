import numpy as np
import matplotlib.pyplot as plt

def euler_step(f,tspan,x0,deltat_max):
    '''
    Euler method function to solve x'=f(x,t) 
    Parameters:
    f     = function of two variables f = f(t,x) that defines the
            right-hand side of the ODE to be solved: f' = f(t,x)
    tspan = the time over which to integrate
    x0    = initial condtion vector x(t0) = x0
    N     = number of steps to split integration time into

    '''

    t = np.zeros(deltat_max + 1)
    x = np.zeros(deltat_max + 1)

    x[0] = x0
    t[0] = 0
    h = tspan/deltat_max
    for k in range(deltat_max):
        t[k + 1] = t[k] + h
        x[k + 1] = x[k] + h * f(x[k], t[k])
    return x,t

if __name__ == "__main__":

    def f(x,t):
        return x

    def solve_to():
        x,t = euler_step(f, tspan = 1, x0=1, deltat_max=10)
        
        


