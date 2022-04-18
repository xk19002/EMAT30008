import numpy as np
import matplotlib.pyplot as plt
from superclass_solver import forweuler, midpoint, RK4

def f(x,t):
    return x

tsteps = np.linspace(0,3,11)

eu = forweuler(f)
eu.init_conds(x0=1)
x1, t1 = eu.sol_meth(tsteps)
plt.plot(t1,x1,label='Forward Euler')

mid = midpoint(f)
mid.init_conds(x0=1)
x2,t2 = mid.sol_meth(tsteps)
plt.plot(t2,x2,label='Midpoint')

rk4 = RK4(f)
rk4.init_conds(x0=1)
x3,t3 = rk4.sol_meth(tsteps)
plt.plot(t3,x3,label='RungeKutta 4')

exact_sol = np.linspace(0,3,301)
plt.plot(exact_sol,np.exp(exact_sol), label='Exact solution')

plt.legend()
plt.show()
