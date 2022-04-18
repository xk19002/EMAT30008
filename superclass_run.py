import numpy as np
import matplotlib.pyplot as plt
from superclass_solver import forweuler, midpoint, RK4

def f(x,t):
    u1,u2 = x
    beta = 2
    return [beta*u1 - u2 -u1*(u1**2 + u2**2), u1 + beta*u2 - u2*(u1**2 + u2**2)]

u1 = 1
u2 = 1
x0 = [u1,u2]
tsteps = np.linspace(0,1,101)

eu = forweuler(f)
eu.init_conds(x0)
x1, t1 = eu.sol_meth(tsteps)
u1 = x1[:,0]
u2 = x1[:,1]
plt.plot(u1,u2,label='Forward Euler')

mid = midpoint(f)
mid.init_conds(x0)
x2,t2 = mid.sol_meth(tsteps)
u1 = x2[:,0]
u2 = x2[:,1]
plt.plot(u1,u2,label='Midpoint')

rk4 = RK4(f)
rk4.init_conds(x0)
x3,t3 = rk4.sol_meth(tsteps)
u1 = x3[:,0]
u2 = x3[:,1]
plt.plot(u1,u2,label='RungeKutta 4')

#exact_sol = np.linspace(0,3,301)
#plt.plot(exact_sol,np.exp(exact_sol), label='Exact solution')

plt.legend()
plt.show()
