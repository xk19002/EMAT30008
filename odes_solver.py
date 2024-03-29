import numpy as np
from matplotlib import pyplot as plt

def euler_scheme(func,x0,tspan,deltat_max):
    '''
    Euler method function to solve x'=f(x,t) 
    Parameters:
    f          = function of two variables f = f(t,x) that defines the
                 right-hand side of the ODE to be solved: f' = f(t,x)    
    tspan      = the time over which to integrate
    x0         = initial condtion vector x(t0) = x0
    deltat_max = number of steps to split integration time into

    '''
    
    f = lambda x,t: np.asarray(func(x,t))
    t = np.zeros(deltat_max + 1)
    if isinstance(x0, (int, float)):
        x = np.zeros(deltat_max + 1)
    else:
        eqs = len(x0)
        x = np.zeros((deltat_max+1, eqs))

    x[0,:] = x0
    t[0] = 0
    h = tspan/deltat_max

    for k in range(deltat_max):
        t[k+1] = t[k] + h
        x[k+1,:] = x[k,:] +h * f(x[k,:], t[k])
    return x, t

if __name__ == "__main__":
    def f(x,t):
        return [x[1], -x[0]]

    x0 = [0,1]
    tspan = 8*np.pi
    for deltat_max in [200,500,1000]:
        x,t =euler_scheme(f,x0,tspan,deltat_max)
        plt.plot(t,x[:,0], linestyle='dashed', label=f'step_size={deltat_max}')

    plt.plot(t, np.sin(t))
    plt.legend()
    plt.show()

    #for deltat_max in [5,10,20,40,100]:
        #x,t = euler_scheme(f, x0=1, tspan=4, deltat_max=deltat_max)
        #plt.plot(t,x, linestyle='dashed', label=f"step_size={deltat_max}")

    #exact = np.linspace(0,4,1001)
    #plt.plot(exact, np.exp(exact), label='Exact')
    #plt.legend()
    #plt.show()


    
