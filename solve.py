import numpy as np
from euler_rk4_solver import euler_step, RK4_step
from matplotlib import pyplot as plt

def f(x,t):
    return -x

tspan = 3
step_size = 0.5
tstep_num = int(round(tspan/step_size)) 
tstep = np.linspace(0, tspan, tstep_num+1)

for method in [RK4_step, euler_step]:
    solve = method(f)
    solve.init_conds(1)
    x,t = solve.solve_to(tstep)
    plt.plot(t,x, label=method.__name__)

exact = np.linspace(0, tspan, 1001)
plt.plot(exact, np.exp(-exact), label='Exact')
plt.legend()
plt.show()
