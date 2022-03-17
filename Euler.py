import numpy as np
import matplotlib.pyplot as plt

def euler_step(f,x0,t0,h):
    '''
    Euler method function to solve x'=f(x,t) 
    Parameters:
    f     = function of two variables f = f(t,x) that defines the
            right-hand side of the ODE to be solved: f' = f(t,x)
    tspan = the time over which to integrate
    x0    = initial condtion vector x(t0) = x0
    N     = number of steps to split integration time into

    '''

    x = x0
    t = t0
    t = t + h
    x = x + h * f(x, t) 
    print(x,t)
    return x,t
    

if __name__ == "__main__":

    def f(x,t):
        return x

    def exact_sol():
        exact = np.linspace(0,1,1001)
        plt.plot(exact, np.exp(exact), label='Exact')
        plt.legend()

    def solve_to(tspan,deltat_max):
        x,t = euler_step(f, x0=1, t0=0, h=tspan/deltat_max)
        print(x,t)


    def solve_ode(x0,t0,tspan):
        
        for deltat_max in [5,10,20]:
        
            t = np.zeros(deltat_max + 1)
            x = np.zeros(deltat_max + 1)

            x[0] = x0
            t[0] = t0
            h = tspan/deltat_max
            
            for k in range(deltat_max):
                t[k + 1] = t[k] + h
                x[k + 1] = x[k] + h * f(x[k], t[k])
            
            x,t = solve_to(tspan=tspan,deltat_max=deltat_max)
            print(x,t)
            plt.plot(t,x, linestyle='dashed', label=f"n={deltat_max}")
            plt.legend()
 

    euler_step(f, x0=1, t0=0, h=1)
    exact_sol()
    solve_to(tspan=1,deltat_max=10) 
    solve_ode(x0=1, t0=0, tspan=10)  
    plt.show()   
    



